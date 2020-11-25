import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# グリッドサーチ系変数
SEED = 42#乱数シード
CV_NUM = 5#クロスバリデーションの分割数
EARLY_STOPPING_ROUNDS=50#評価指標がこの回数連続で改善しなくなった時点で学習をストップ
SCORING='r2'#グリッドサーチで最大化する評価指標
#グリッドサーチ用パラメータ(https://qiita.com/R1ck29/items/50ba7fa5afa49e334a8f)
CV_PARAMS = {'eval_metric':['rmse'],#データの評価指標
             'objective':['reg:squarederror'],#最小化させるべき損失関数
             'random_state':[SEED],#乱数シード
             'booster': ['gbtree'],
             'learning_rate':[0.1,0.3,0.5],
             'min_child_weight':[1,5,15],
             'max_depth':[3,5,7],
             'colsample_bytree':[0.5,0.8,1.0],
             'subsample':[0.5,0.8,1.0]
            }


class XGBTuning():
    #初期化（pandasではなくndarrayを入力）
    def __init__(self, X, y, X_colnames, y_colname=None):
        self.X = X
        self.y = y
        self.X_colnames = X_colnames
        self.y_colname = y_colname
        self.cv_params = None

    #グリッドサーチ＋クロスバリデーション
    def grid_search_tuning(self, cv_params=CV_PARAMS, cv_num=CV_NUM, seed=SEED, early_stopping_rounds=EARLY_STOPPING_ROUNDS):
        #パラメータを保持
        self.cv_params=cv_params
        #XGBoostのインスタンス作成
        cv_model = xgb.XGBRegressor()
        # グリッドサーチのインスタンス作成
        # n_jobs=-1にするとCPU100%で全コア並列計算。とても速い。
        cv = GridSearchCV(cv_model, cv_params, cv = cv_num, scoring= SCORING, n_jobs =-1)
        
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

    def get_cv_params(self):
        return self.cv_params