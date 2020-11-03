#%%読込
import pandas as pd
from custom_pair_plot import CustomPairPlot

#使用するフィールド
KEY_VALUE = 'ward_before'#キー列
OBJECTIVE_VARIALBLE = 'approval_rate'#目的変数
EXPLANATORY_VALIABLES = ['1_over60','2_between_30to60','3_male_ratio','4_required_time','5_household_member','6_income']
USE_EXPLANATORY = ['2_between_30to60','3_male_ratio','4_required_time','5_household_member']

df = pd.read_csv(f'./osaka_metropolis_english.csv')
df
# %%pair_analyzer
use_cols = [OBJECTIVE_VARIALBLE] + EXPLANATORY_VALIABLES
gp = CustomPairPlot()
#gp.pairanalyzer(df[use_cols])
# %%pair_analyzer(特徴量削減)
use_cols = [OBJECTIVE_VARIALBLE] + USE_EXPLANATORY
#gp.pairanalyzer(df[use_cols])
# %%XGBoost用設定読込
import xgboost as xgb
from sklearn import metrics as met
import sklearn as skl
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

#目的変数と説明変数を取得（pandasではなくndarrayに変換）
y = df[[OBJECTIVE_VARIALBLE]].values
X = df[USE_EXPLANATORY].values
#グリッドサーチと性能評価の共通パラメータ
num_round=10000#最大学習回数
early_stopping_rounds=50#評価指標がこの回数連続で改善しなくなった時点で学習をストップ
seed = 42#乱数シード

#%%グリッドサーチによるパラメータ最適化

#グリッドサーチ用パラメータ(https://qiita.com/R1ck29/items/50ba7fa5afa49e334a8f)
cv_params = {'eval_metric':['rmse'],#データの評価指標
             'objective':['reg:squarederror'],#最小化させるべき損失関数
             'random_state':[seed],#乱数シード
             'booster': ['gbtree'],
             'learning_rate':[0.1,0.3,0.5],
             'min_child_weight':[1,5,15],
             'max_depth':[3,5,7],
             'colsample_bytree':[0.5,0.8,1.0],
             'subsample':[0.5,0.8,1.0]
            }

#XGBoostのインスタンス作成
cv_model = xgb.XGBRegressor()
#グリッドサーチのインスタンス作成
# n_jobs=-1にするとCPU100%で全コア並列計算。とても速い。
cv = GridSearchCV(cv_model, cv_params, cv = 5, scoring= 'r2', n_jobs =-1)

#学習とテストデータに分割
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#グリッドサーチ実行
evallist = [(X, y)]
cv.fit(X,
        y,
        eval_set=evallist,
        early_stopping_rounds=early_stopping_rounds
        )

#最適パラメータの表示
print('最適パラメータ ' + str(cv.best_params_))
print('変数重要度' + str(cv.best_estimator_.feature_importances_))

#%%性能評価(Leave-One-Out)
#パラメータにグリッドサーチでの最適パラメータを使用
params = cv.best_params_
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
        f.write('\n最適パラメータ' + str(cv.best_params_))
        f.write('\nグリッドサーチ対象' + str(cv_params))
        f.write('\n変数重要度' + str(cv.best_estimator_.feature_importances_))
        f.write('\nRMSE平均' + str(df_result['eval_rmse_min'].mean()))
        f.write('\n相関係数' + str(df_result[['pred_value','real_value']].corr().iloc[1,0]))
        f.write('\n予測誤差の最大値' + str(max((df_result['pred_value'] - df_result['real_value']).abs())))

# %%
