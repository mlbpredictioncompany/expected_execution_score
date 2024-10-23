import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import d2_log_loss_score
import joblib

######

# READ DATA
df = pd.read_parquet('ab_execution.parquet')
print(df)


# PARSE FIELDS
df['batter_score_differential'] = np.where(df['half_inning'] == 'top', df['away_score'] - df['home_score'], df['home_score'] - df['away_score'])
df['runs_scored'] = np.where(df['half_inning'] == 'top', df['next_away_score'] - df['away_score'], df['next_home_score'] - df['home_score'])
df['in_play'] = np.where(df['effective_batted_ball_exit_velocity'].isna(), 0, 1)
print(df)


# DROP INVALID
df['delta'] = 1 - (df['next_runner_1b_binary'] - df['runner_1b_binary']) - (df['next_runner_2b_binary'] - df['runner_2b_binary']) - (df['next_runner_3b_binary'] - df['runner_3b_binary']) - df['runs_scored'] - (df['next_outs'] - df['outs'])
df = df.loc[(df['delta'] == 0) | ((df['delta'] > 0) & (df['next_outs'] == 0))].copy()
print(df)


# STATE IDS
df['start_state_str'] = df[['outs', 'runner_1b_binary', 'runner_2b_binary', 'runner_3b_binary']].astype(str).agg('-'.join, axis=1)
df['next_state_str'] = df[['next_outs', 'next_runner_1b_binary', 'next_runner_2b_binary', 'next_runner_3b_binary', 'runs_scored']].astype(str).agg('-'.join, axis=1)
df['actual_next_state_str'] = df['next_state_str']
print(df)


# MODEL OBJECT
class StateModel:
    def __init__(self):
        self.scaler = None
        self.poly = None
        self.labels = {}
        self.models = {}
        self.calibrators = {}
        self.meta = ['start_state_str']
        self.output = ['next_state_str']
        self.features = ['inning', 'batter_score_differential', 'effective_batted_ball_exit_velocity', 'effective_batted_ball_launch_angle', 'effective_batted_ball_spray_angle']
        self.poly_features = ['effective_batted_ball_exit_velocity', 'effective_batted_ball_launch_angle', 'effective_batted_ball_spray_angle']
        self.poly_feature_names = []
        self.all_features = []
        self.out_columns = ['game_date', 'game_id', 'ab_id', 'pitcher_id', 'pitcher_name', 'batter_id', 'batter_name', 'inning', 'half_inning', 'outs', 'runner_1b_binary', 'runner_2b_binary', 'runner_3b_binary', 'home_score', 'away_score', 'start_state_str', 'actual_next_state_str', 'in_play']


    def fit(self, df):
        # KEEP ONLY IN-PLAY
        df_train = df[self.meta + self.features + self.output].dropna().copy()

        # ITERATE OVER STATES
        for start_state in df_train['start_state_str'].unique():
            # SELECT SUBSET
            df_i = df_train.loc[df_train['start_state_str'] == start_state].copy()

            # SELECT X
            x_i = df_i[self.features].values

            # LABEL ENCODE Y
            self.labels[start_state] = LabelEncoder()
            self.labels[start_state].fit(df_i[self.output].values.ravel())
            y_i = self.labels[start_state].transform(df_i[self.output].values.ravel())

            # TRAIN MODEL
            self.models[start_state] = Pipeline([
                ('scaler_pre', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
                ('scaler_post', StandardScaler()),
                ('model', LogisticRegression(max_iter=10000, class_weight='balanced')),
            ])
            self.models[start_state].fit(x_i, y_i)
            print(f'Model Fit for State {start_state}')

            # TRAIN CALIBRATOR
            self.calibrators[start_state] = LogisticRegression(max_iter=10000)
            self.calibrators[start_state].fit(self.models[start_state].predict_proba(x_i), y_i)
            print(f'Calibrator Fit for State {start_state}')

        return


    def predict(self, df):
        # SEGMENT IN-PLAY ROWS
        df_in_play = df.loc[~df[self.features].isna().any(axis=1)].copy()
        df_fixed = df.loc[df[self.features].isna().any(axis=1)].copy()

        # ITERATE OVER IN-PLAY
        df_in_play_list = []

        # ITERATE OVER STATES
        for start_state in df_in_play['start_state_str'].unique():
            # SELECT SUBSET
            df_i = df_in_play.loc[df_in_play['start_state_str'] == start_state].copy()

            # SELECT X
            x_i = df_i[self.features].values

            # PREDICT MODEL
            probs = self.models[start_state].predict_proba(x_i)

            # APPLY CALIBRATION
            probs = self.calibrators[start_state].predict_proba(probs)

            # ASSIGN TO DF
            pred_labels = np.arange(probs.shape[1])
            df_i[pred_labels] = probs

            # MELT
            df_i = df_i.melt(id_vars=self.out_columns, value_vars=pred_labels, var_name='pred_label', value_name='probability')

            # CONVERT STATE NAMES
            df_i['true_label'] = self.labels[start_state].transform(df_i['actual_next_state_str'])
            df_i['next_state_str'] = self.labels[start_state].inverse_transform(df_i['pred_label'].astype(int))

            # APPEND LIST
            df_in_play_list.append(df_i)

        # CONCAT IN-PLAY
        df_in_play = pd.concat(df_in_play_list)

        # ASSIGN FIXED VALUES
        df_fixed['probability'] = 1.0
        df_fixed = df_fixed[self.out_columns + ['next_state_str', 'probability']]

        # AGGREGATE
        df_out = pd.concat([df_in_play, df_fixed], axis=0)

        return df_out



# CREATE/FIT STATE MODEL
state_model = StateModel()
state_model.fit(df)


# SAVE/LOAD STATE MODEL
state_model_filename = 'StateModel.pkl'

joblib.dump(state_model, state_model_filename) # SAVE
state_model = joblib.load(state_model_filename) # LOAD


# PREDICT USING STATE MODEL
df_pred = state_model.predict(df)
print(df_pred)


# EVALUATE CALIBRATION
df_pred['actual_prob'] = np.where(df_pred['next_state_str'] == df_pred['actual_next_state_str'], 1, 0)

calibration = df_pred.groupby(['start_state_str', 'next_state_str', 'in_play'])[['probability', 'actual_prob']].sum().reset_index()
calibration['probability'] = calibration['probability'] / calibration.groupby(['start_state_str', 'in_play'])['probability'].transform('sum')
calibration['actual_prob'] = calibration['actual_prob'] / calibration.groupby(['start_state_str', 'in_play'])['actual_prob'].transform('sum')
calibration['delta'] = calibration['probability'] - calibration['actual_prob']
calibration.sort_values('delta', inplace=True)
print(calibration)


# EVALUATE MODEL PERFORMANCE        
pivot = pd.pivot_table(data=df_pred.loc[df_pred['in_play'] == 1], index=['start_state_str', 'ab_id', 'true_label'], columns=['pred_label'], values='probability', fill_value=0).reset_index()
print(pivot)

d2 = pivot.groupby('start_state_str').apply(lambda x: d2_log_loss_score(y_true=x['true_label'], y_pred=x[np.arange(x['true_label'].max() + 1)]), include_groups=False)
print(d2)


# PARSE NEXT STATE
df_pred[['next_outs', 'next_runner_1b_binary', 'next_runner_2b_binary', 'next_runner_3b_binary', 'runs_scored']] = df_pred['next_state_str'].str.split('-', expand=True).astype(int)
print(df_pred)


# DROP INVALID
df_pred['delta'] = 1 - (df_pred['next_runner_1b_binary'] - df_pred['runner_1b_binary']) - (df_pred['next_runner_2b_binary'] - df_pred['runner_2b_binary']) - (df_pred['next_runner_3b_binary'] - df_pred['runner_3b_binary']) - df_pred['runs_scored'] - (df_pred['next_outs'] - df_pred['outs'])
df_pred = df_pred.loc[(df_pred['delta'] == 0) | ((df_pred['delta'] > 0) & (df_pred['next_outs'] == 0))].copy()
print(df_pred)


# GET NEXT STATE
df_pred['inning_end'] = np.where(df_pred['delta'] > 0, 1, 0)
df_pred['opp_half_inning'] = np.where(df_pred['half_inning'] == 'top', 'bottom', 'top')
df_pred['next_half_inning'] = np.where(df_pred['inning_end'] == 1, df_pred['opp_half_inning'], df_pred['half_inning'])
df_pred['next_inning'] = np.where((df_pred['inning_end'] == 1) & (df_pred['half_inning'] == 'bottom'), df_pred['inning'] + 1, df_pred['inning'])
df_pred['next_home_score'] = np.where(df_pred['half_inning'] == 'bottom', df_pred['home_score'] + df_pred['runs_scored'], df_pred['home_score'])
df_pred['next_away_score'] = np.where(df_pred['half_inning'] == 'top', df_pred['away_score'] + df_pred['runs_scored'], df_pred['away_score'])
print(df_pred)


# READ EXECUTION SCORES
execution_scores = pd.read_parquet('execution_score_uninformed.parquet')

execution_scores = execution_scores[['half_inning', 'inning', 'outs', 'runner_1b', 'runner_2b', 'runner_3b', 'home_score', 'away_score', 'next_half_inning', 'next_inning', 'next_outs', 'next_runner_1b', 'next_runner_2b', 'next_runner_3b', 'next_home_score', 'next_away_score', 'z_score']]
execution_scores.columns = ['half_inning', 'inning', 'outs', 'runner_1b_binary', 'runner_2b_binary', 'runner_3b_binary', 'home_score', 'away_score', 'next_half_inning', 'next_inning', 'next_outs', 'next_runner_1b_binary', 'next_runner_2b_binary', 'next_runner_3b_binary', 'next_home_score', 'next_away_score', 'z_score']
execution_scores.set_index(['half_inning', 'inning', 'outs', 'runner_1b_binary', 'runner_2b_binary', 'runner_3b_binary', 'home_score', 'away_score', 'next_half_inning', 'next_inning', 'next_outs', 'next_runner_1b_binary', 'next_runner_2b_binary', 'next_runner_3b_binary', 'next_home_score', 'next_away_score'], inplace=True)
print(execution_scores)


# JOIN EXECUTION SCORES
df_pred = df_pred.join(execution_scores, how='left', on=['half_inning', 'inning', 'outs', 'runner_1b_binary', 'runner_2b_binary', 'runner_3b_binary', 'home_score', 'away_score', 'next_half_inning', 'next_inning', 'next_outs', 'next_runner_1b_binary', 'next_runner_2b_binary', 'next_runner_3b_binary', 'next_home_score', 'next_away_score'])
print(df_pred)


# AGGREGATE SCORES
df_pred['expected_score'] = df_pred['z_score'] * df_pred['probability']
df_pred = df_pred.groupby('ab_id')[['expected_score', 'probability']].sum().reset_index()
df_pred['expected_score'] = df_pred['expected_score'] / df_pred['probability']
print(df_pred)


# JOIN BACK TO ORIGINAL
df = df.merge(df_pred[['ab_id', 'expected_score']], how='left', on=['ab_id'])
print(df)
