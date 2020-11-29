#%%読込
import pandas as pd
from custom_pair_plot import CustomPairPlot
import seaborn as sns

#パラメータ最適化の手法(Grid, Random, Bayes, Optuna)
PARAM_TUNING_METHOD = 'Random'

#使用するフィールド
KEY_VALUE = 'ward_before'#キー列
OBJECTIVE_VARIALBLE = 'approval_rate'#目的変数
EXPLANATORY_VALIABLES = ['1_over60','2_between_30to60','3_male_ratio','4_required_time','5_household_member','6_income']#説明変数
USE_EXPLANATORY = ['2_between_30to60','3_male_ratio','5_household_member','latitude']#使用する説明変数
#データ読込
df = pd.read_csv(f'./osaka_metropolis_english.csv')
df
# %%1. pair_analyzerでデータの可視化
use_cols = [OBJECTIVE_VARIALBLE] + EXPLANATORY_VALIABLES
gp = CustomPairPlot()
#gp.pairanalyzer(df[use_cols])
# %%1. pair_analyzerでデータの可視化(特徴量削減後)
use_cols = [OBJECTIVE_VARIALBLE] + USE_EXPLANATORY
#gp.pairanalyzer(df[use_cols])

# %% XGBoost＆パラメータ最適化用設定読込
import xgboost as xgb
from sklearn import metrics as met
import sklearn as skl
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from xgb_param_tuning import XGBRegressorTuning

#目的変数と説明変数を取得（pandasではなくndarrayに変換）
y = df[[OBJECTIVE_VARIALBLE]].values
X = df[USE_EXPLANATORY].values
#グリッドサーチと性能評価の共通パラメータ
num_round=10000#最大学習回数
early_stopping_rounds=50#評価指標がこの回数連続で改善しなくなった時点で学習をストップ
seed = 43#乱数シード

#グリッドサーチによるパラメータ最適化メソッド
def grid_search(X, y):
    #パラメータ最適化クラス
    xgb_tuning = XGBRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
    #グリッドサーチ実行
    cv = xgb_tuning.grid_search_tuning()
    tuning_params = xgb_tuning.get_tuning_params()#グリッドサーチに使用したパラメータ
    return cv, tuning_params

#ランダムサーチによるパラメータ最適化メソッド
def random_search(X, y):
    #パラメータ最適化クラス
    xgb_tuning = XGBRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
    #ランダムサーチ実行
    cv = xgb_tuning.random_search_tuning()
    tuning_params = xgb_tuning.get_tuning_params()#ランダムサーチに使用したパラメータ
    return cv, tuning_params

#%%2. パラメータ最適化
if PARAM_TUNING_METHOD == 'Grid':
    cv, tuning_params = grid_search(X, y)
elif PARAM_TUNING_METHOD == 'Random':
    cv, tuning_params = random_search(X, y)

params = cv.best_params_#最適化したパラメータを保持
feature_importances = cv.best_estimator_.feature_importances_#特徴量重要度

#%%3. 性能評価(Leave-One-Out)
#結果保持用のDataFrame
df_result = pd.DataFrame(columns=['test_index','eval_rmse_min','train_rmse_min','num_train'])

#Leave-One-Outでデータ分割して性能評価
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):#全データに対して分割ループ
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    dtrain = xgb.DMatrix(X_train, label=y_train)#学習用データ
    dtest = xgb.DMatrix(X_test, label=y_test)#テストデータ
    evals = [(dtest, 'eval'),(dtrain, 'train')]#結果表示用の学習データとテストデータを指定
    evals_result = {}#結果保持用

    #学習実行
    model = xgb.train(params,
                    dtrain,#訓練データ
                    num_boost_round=num_round,
                    early_stopping_rounds=early_stopping_rounds,
                    evals=evals,
                    evals_result=evals_result
                    )
    
    #モデルの性能評価
    test_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    num_train = len(evals_result['eval']['rmse'])
    df_result = df_result.append({'test_index': test_index[0],
                    'key_value': df[[KEY_VALUE]].iloc[test_index[0],0],
                    'pred_value': test_pred[0],
                    'real_value': df[[OBJECTIVE_VARIALBLE]].iloc[test_index[0],0],
                    'eval_rmse_min': evals_result['eval']['rmse'][num_train - 1],
                    'train_rmse_min': evals_result['train']['rmse'][num_train - 1],
                    'num_train': num_train},
                    ignore_index=True)

#性能評価結果の表示
print('RMSE平均' + str(df_result['eval_rmse_min'].mean()))
print('相関係数' + str(df_result[['pred_value','real_value']].corr().iloc[1,0]))
print('予測誤差の最大値' + str(max((df_result['pred_value'] - df_result['real_value']).abs())))

#結果を出力
dt_now = datetime.now().strftime('%Y%m%d%H%M%S')
feat_use = 'feat' + '-'.join([ex.split('_')[0] for ex in USE_EXPLANATORY])
#評価結果
df_result.to_csv(f"{os.getenv('HOMEDRIVE')}{os.getenv('HOMEPATH')}\Desktop\{feat_use}_{dt_now}_result.csv")

path = f"{os.getenv('HOMEDRIVE')}{os.getenv('HOMEPATH')}\Desktop\{feat_use}_{dt_now}_result.txt"
with open(path, mode='w') as f:
        f.write('特徴量' + str(USE_EXPLANATORY))
        f.write('\n最適化手法' + PARAM_TUNING_METHOD)
        f.write('\n最適パラメータ' + str(params))
        f.write('\nグリッドサーチ対象' + str(tuning_params))
        f.write('\n変数重要度' + str(feature_importances))
        f.write('\nRMSE平均' + str(df_result['eval_rmse_min'].mean()))
        f.write('\n相関係数' + str(df_result[['pred_value','real_value']].corr().iloc[1,0]))
        f.write('\n予測誤差の最大値' + str(max((df_result['pred_value'] - df_result['real_value']).abs())))

#散布図表示
sns.regplot(x="pred_value", y="real_value", data=df_result, ci=0)

# %%前回と今回の比較散布図
sns.regplot(x="approval_former", y="approval_rate", data=df, ci=0)
# %%
