from xgb_param_tuning import XGBRegressorTuning
import pandas as pd
from datetime import datetime
import os

# パラメータ最適化の手法(Grid, Random, Bayes, Optuna)
PARAM_TUNING_METHODS = ['Bayes']
# 最適化で使用する乱数シード一覧
SEEDS = [42, 43, 44, 45]

#使用するフィールド
KEY_VALUE = 'ward_before'#キー列
OBJECTIVE_VARIALBLE = 'approval_rate'#目的変数
EXPLANATORY_VALIABLES = ['1_over60','2_between_30to60','3_male_ratio','4_required_time','5_household_member','6_income']#説明変数
USE_EXPLANATORY = ['2_between_30to60','3_male_ratio','5_household_member','latitude']#使用する説明変数


# 現在時刻
dt_now = datetime.now().strftime('%Y%m%d%H%M%S')
#データ読込
df = pd.read_csv(f'./osaka_metropolis_english.csv')
#目的変数と説明変数を取得（pandasではなくndarrayに変換）
y = df[[OBJECTIVE_VARIALBLE]].values
X = df[USE_EXPLANATORY].values

#パラメータ最適化クラス
xgb_tuning = XGBRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)

# 手法を変えて最適化
for method in PARAM_TUNING_METHODS:
    # 乱数を変えて最適化をループ実行
    df_result_seeds = xgb_tuning.tuning_multiple_seeds(method, seeds=SEEDS)
    df_result_seeds.to_csv(f"{os.getenv('HOMEDRIVE')}{os.getenv('HOMEPATH')}\Desktop\{method}_{dt_now}_result.csv", index=False)
