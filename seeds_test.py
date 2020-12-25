from xgb_param_tuning import XGBRegressorTuning
from xgb_validation import XGBRegressorValidation
import pandas as pd
from datetime import datetime
import os
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import numpy as np

# 結果出力先
OUTPUT_DIR = f"{os.getenv('HOMEDRIVE')}{os.getenv('HOMEPATH')}\Desktop"
# 最適化で最大化する評価指標('r2', 'neg_mean_squared_error', 'neg_mean_squared_log_error')
SCORING = 'r2'
# パラメータ最適化の手法(Grid, Random, Bayes, Optuna)
PARAM_TUNING_METHODS = ['Bayes']
# 最適化で使用する乱数シード一覧
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

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

# パラメータ最適化クラス
xgb_tuning = XGBRegressorTuning(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)
# 検証用クラス
xgb_validation = XGBRegressorValidation(X, y, USE_EXPLANATORY, y_colname=OBJECTIVE_VARIALBLE)

# 全乱数シードで算出したパラメータの平均値使用(基本的にはこのメソッドは不使用)
def calc_params_mean(df_params):
    # int型、float型、それ以外に分けて平均値算出
    df_int = df_params.select_dtypes(include=[int, 'int64', 'int32'])
    df_float = df_params.select_dtypes(include=float)
    df_str = df_params.select_dtypes(exclude=[int, 'int64', 'int32', float])
    df_int = df_int.mean().apply(lambda x: int(x))
    df_float = df_float.mean()
    df_str = df_str.iloc[0, :]
    df_params = pd.concat([df_int, df_float, df_str])
    # dict化
    return df_params.to_dict()

# 乱数シードごとに別々のパラメータを使用
def convert_params_list(df_params):
    params_list = []
    for i in range(len(df_params)):
        params_list.append(df_params.iloc[i, :].to_dict())
    return params_list

# 手法を変えて最適化
for method in PARAM_TUNING_METHODS:
    # 乱数を変えて最適化をループ実行
    df_result_seeds, param_range = xgb_tuning.multiple_seeds_tuning(method, seeds=SEEDS, scoring=SCORING)
    # 結果出力
    df_result_seeds.to_csv(f"{OUTPUT_DIR}\{method}_seed{'-'.join([str(s) for s in SEEDS])}_tuning_{dt_now}.csv", index=False)
    param_range_path = f"{OUTPUT_DIR}\{method}_seed{'-'.join([str(s) for s in SEEDS])}_param_range_{dt_now}.txt"
    with open(param_range_path, mode='w') as f:
        f.write('{')
        for k, v in param_range.items():
            f.write(str(k) + ':' + str(v) + ',\n')
        f.write('}')

    # パラメータ記載列（'best_'で始まる列）のみ抽出
    extractcols = df_result_seeds.columns.str.startswith('best_')
    df_params = df_result_seeds.iloc[:, extractcols]
    # 列名から'best_'を削除
    for colname in df_params.columns:
        df_params = df_params.rename(columns={colname:colname.replace('best_', '')})
    
    # 全乱数シードで算出したパラメータの平均値使用
    #params = calc_param_mean(df_params)
    params = convert_params_list(df_params)

    # 最適化したモデルを検証
    validation_score, validation_detail = xgb_validation.multiple_seeds_validation(params, seeds=SEEDS, method='leave_one_out')
    validation_score.to_csv(f"{OUTPUT_DIR}\{method}_seed{'-'.join([str(s) for s in SEEDS])}_valid_score_{dt_now}.csv", index=False)
    validation_detail.to_csv(f"{OUTPUT_DIR}\{method}_seed{'-'.join([str(s) for s in SEEDS])}_valid_detail_{dt_now}.csv", index=False)