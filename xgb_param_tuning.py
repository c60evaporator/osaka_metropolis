import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

# 共通変数
SEED = 42  # 乱数シード
BOOSTER = 'gbtree'  # ブースター('gbtree':ツリーモデル, 'dart':ツリーモデル, 'gblinesr':線形モデル)
OBJECTIVE = 'reg:squarederror'  # 最小化させるべき損失関数(デフォルト:'reg:squarederror')
EVAL_METRIC = 'rmse'  # データの評価指標。基本的にはOBJECTIVEと1対1対応(デフォルト:'rmse')
CV_NUM = 5  #クロスバリデーションの分割数
EARLY_STOPPING_ROUNDS=50  # 評価指標がこの回数連続で改善しなくなった時点で学習をストップ
SCORING='r2'  # グリッドサーチで最大化する評価指標
N_ITER = 150  # ランダムサーチorベイズ最適化の繰り返し回数

# グリッドサーチ用パラメータ(https://qiita.com/R1ck29/items/50ba7fa5afa49e334a8f)
CV_PARAMS_GRID = {'eval_metric':[EVAL_METRIC],#データの評価指標
             'objective':[OBJECTIVE],#最小化させるべき損失関数
             'random_state':[SEED],#乱数シード
             'booster': [BOOSTER],#ブースター
             'learning_rate':[0.1,0.3,0.5],#過学習のバランス(高いほど過学習寄り、低いほど汎化寄り)
             'min_child_weight':[1,5,15],#葉に割り当てるスコアwiの合計の最小値。これを下回った場合、それ以上の分割を行わない
             'max_depth':[3,5,7],#木の深さの最大値
             'colsample_bytree':[0.5,0.8,1.0],#列のサブサンプリングを行う比率
             'subsample':[0.5,0.8,1.0]#木を構築する前にデータのサブサンプリングを行う比率。1 なら全データ使用、0.5なら半分のデータ使用
            }

# ランダムサーチ用パラメータ
CV_PARAMS_RANDOM = {'eval_metric':[EVAL_METRIC],#データの評価指標
             'objective':[OBJECTIVE],#最小化させるべき損失関数
             'random_state':[SEED],#乱数シード
             'booster': [BOOSTER],
             'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6],
             'min_child_weight':[1,3,5,7,11,15,19,25],
             'max_depth':[3,4,5,6,7],
             'colsample_bytree':[0.4,0.5,0.6,0.7,0.8,0.9,1.0],
             'subsample':[0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            }

# ベイズ最適化用パラメータ
BAYES_PARAMS = {'learning_rate': (0.1,1),
             'min_child_weight': (1,30),
             'max_depth': (1,20),
             'colsample_bytree': (0.1,1),
             'subsample': (0.1,1)
            }

class XGBRegressorTuning():
    #初期化（pandasではなくndarrayを入力）
    def __init__(self, X, y, X_colnames, y_colname=None):
        self.X = X
        self.y = y
        self.X_colnames = X_colnames
        self.y_colname = y_colname
        self.tuning_params = None
        self.seed = None
        self.early_stopping_rounds = None

    #グリッドサーチ＋クロスバリデーション
    def grid_search_tuning(self, cv_params=CV_PARAMS_GRID, cv_num=CV_NUM, seed=SEED, early_stopping_rounds=EARLY_STOPPING_ROUNDS):
        #引数を反映
        cv_params['random_state'] = [seed]
        self.tuning_params=cv_params
        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds
        #XGBoostのインスタンス作成
        cv_model = xgb.XGBRegressor()
        # グリッドサーチのインスタンス作成
        # n_jobs=-1にするとCPU100%で全コア並列計算。とても速い。
        cv = GridSearchCV(cv_model, cv_params, cv=cv_num, scoring= SCORING, n_jobs =-1)
        
        #グリッドサーチ実行
        evallist = [(self.X, self.y)]
        cv.fit(self.X,
                self.y,
                eval_set=evallist,
                early_stopping_rounds=early_stopping_rounds
                )

        #最適パラメータの表示
        print('最適パラメータ ' + str(cv.best_params_))
        print('変数重要度' + str(cv.best_estimator_.feature_importances_))

        #特徴量重要度の描画
        features = list(reversed(self.X_colnames))
        importances = list(reversed(cv.best_estimator_.feature_importances_.tolist()))
        plt.barh(features,importances)

        #グリッドサーチでの探索結果を返す
        return cv

    #ランダムサーチ＋クロスバリデーション
    def random_search_tuning(self, cv_params=CV_PARAMS_RANDOM, cv_num=CV_NUM, n_iter=N_ITER, seed=SEED, early_stopping_rounds=EARLY_STOPPING_ROUNDS):
        #引数を反映
        cv_params['random_state'] = [seed]
        self.tuning_params=cv_params
        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds
        #XGBoostのインスタンス作成
        cv_model = xgb.XGBRegressor()
        # ランダムサーチのインスタンス作成
        # n_jobs=-1にするとCPU100%で全コア並列計算。とても速い。
        cv = RandomizedSearchCV(cv_model, cv_params, cv = cv_num, random_state=seed, n_iter=n_iter, scoring= SCORING, n_jobs =-1)
        
        #グリッドサーチ実行
        evallist = [(self.X, self.y)]
        cv.fit(self.X,
                self.y,
                eval_set=evallist,
                early_stopping_rounds=early_stopping_rounds
                )

        #最適パラメータの表示
        print('最適パラメータ ' + str(cv.best_params_))
        print('変数重要度' + str(cv.best_estimator_.feature_importances_))

        #特徴量重要度の描画
        features = list(reversed(self.X_colnames))
        importances = list(reversed(cv.best_estimator_.feature_importances_.tolist()))
        plt.barh(features,importances)

        #グリッドサーチでの探索結果を返す
        return cv

    #ベイズ最適化時の評価指標算出メソッド(bayes_optは指標を最大化するので、RMSE等のLower is betterな指標は符号を負にして返す)
    def xgb_reg_evaluate(self, learning_rate, min_child_weight, subsample, colsample_bytree, max_depth):
        params = {'eval_metric': EVAL_METRIC,
                'objective':OBJECTIVE,
                'random_state':self.seed,
                'booster':BOOSTER,
                'learning_rate':learning_rate,              
                'min_child_weight': int(min_child_weight),
                'max_depth': int(max_depth),
                'colsample_bytree': colsample_bytree,
                'subsample': subsample,
                }
        #XGBoostのインスタンス作成
        cv_model = xgb.XGBRegressor()
        cv_model.set_params(**params)
        evallist = [(self.X, self.y)]
        cv_model.fit(self.X,
                self.y,
                eval_set=evallist,
                early_stopping_rounds=self.early_stopping_rounds
                )
        
        pred = cv_model.predict(self.X)
        score = -mean_squared_error(self.y, pred, squared=False)#RMSEのマイナスを評価指標に（squared=TrueだとMSEになるので注意）
        return score

    #ベイズ最適化(bayes_opt)
    def bayes_opt_tuning(self, beyes_params=BAYES_PARAMS, cv_num=CV_NUM, n_iter=N_ITER, seed=SEED, early_stopping_rounds=EARLY_STOPPING_ROUNDS):
        #引数を反映
        beyes_params['random_state'] = [seed]
        self.tuning_params=beyes_params
        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds
        #ベイズ最適化を実行
        xgb_bo = BayesianOptimization(self.xgb_reg_evaluate, beyes_params, random_state=seed)
        xgb_bo.maximize(init_points=15, n_iter=n_iter, acq='ei')

    def get_tuning_params(self):
        return self.tuning_params