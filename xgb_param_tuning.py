import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import time

#回帰パラメータチューニング
class XGBRegressorTuning():
    # 共通定数
    SEED = 45  # 乱数シード
    BOOSTER = 'gbtree'  # ブースター('gbtree':ツリーモデル, 'dart':ツリーモデル, 'gblinesr':線形モデル)
    OBJECTIVE = 'reg:squarederror'  # 最小化させるべき損失関数(デフォルト:'reg:squarederror')
    EVAL_METRIC = 'rmse'  # データの評価指標。基本的にはOBJECTIVEと1対1対応(デフォルト:'rmse')
    CV_NUM = 5  #クロスバリデーションの分割数
    EARLY_STOPPING_ROUNDS=50  # 評価指標がこの回数連続で改善しなくなった時点で学習をストップ
    SCORING='r2'  # 最適化で最大化する評価指標(デフォルト:'r2')
    # TODO: SCORINGにデフォルトでRMSEが存在しないので、sklearn.metrics.make_scorerで自作の必要あり(https://qiita.com/kimisyo/items/afdf76b9b6fcade640ed)
    # (https://qiita.com/taruto1215/items/2b1f7224a9a4f43906d8)
    #最適化対象外パラメータ
    NOT_OPT_PARAMS = {'eval_metric':[EVAL_METRIC],#データの評価指標
                'objective':[OBJECTIVE],#最小化させるべき損失関数
                'random_state':[SEED],#乱数シード
                'booster': [BOOSTER],#ブースター
                }

    # グリッドサーチ用パラメータ(https://qiita.com/R1ck29/items/50ba7fa5afa49e334a8f)
    CV_PARAMS_GRID = {'learning_rate':[0.1,0.3,0.5],#過学習のバランス(高いほど過学習寄り、低いほど汎化寄り)
                'min_child_weight':[1,5,15],#葉に割り当てるスコアwiの合計の最小値。これを下回った場合、それ以上の分割を行わない
                'max_depth':[3,5,7],#木の深さの最大値
                'colsample_bytree':[0.5,0.8,1.0],#列のサブサンプリングを行う比率
                'subsample':[0.5,0.8,1.0]#木を構築する前にデータのサブサンプリングを行う比率。1 なら全データ使用、0.5なら半分のデータ使用
                }
    CV_PARAMS_GRID.update(NOT_OPT_PARAMS)

    # ランダムサーチ用パラメータ
    N_ITER_RANDOM = 150  # ランダムサーチorベイズ最適化の繰り返し回数
    CV_PARAMS_RANDOM = {'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6],
                'min_child_weight':[1,3,5,7,11,15,19,25],
                'max_depth':[3,4,5,6,7],
                'colsample_bytree':[0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                'subsample':[0.4,0.5,0.6,0.7,0.8,0.9,1.0]
                }
    CV_PARAMS_RANDOM.update(NOT_OPT_PARAMS)

    # ベイズ最適化用パラメータ
    N_ITER_BAYES = 75  # ベイズ最適化の繰り返し回数
    INIT_POINTS = 10  # ランダムな探索を何回行うか
    ACQ = 'ei'  # 獲得関数(https://ohke.hateblo.jp/entry/2018/08/04/230000)
    BAYES_PARAMS = {'learning_rate': (0.1,0.8),
                'min_child_weight': (1,20),
                'max_depth': (1,10),
                'colsample_bytree': (0.3,1),
                'subsample': (0.3,1)
                }
    BAYES_NOT_OPT_PARAMS = {k: v[0] for k, v in NOT_OPT_PARAMS.items()}
    
    #初期化（pandasではなくndarrayを入力）
    def __init__(self, X, y, X_colnames, y_colname=None):
        self.X = X
        self.y = y
        self.X_colnames = X_colnames
        self.y_colname = y_colname
        self.tuning_params = None
        self.seed = None
        self.cv_num = None
        self.early_stopping_rounds = None
        self.feature_importances = None
        self.elapsed_time = None

    #グリッドサーチ＋クロスバリデーション
    def grid_search_tuning(self, cv_params=CV_PARAMS_GRID, cv_num=CV_NUM, seed=SEED, early_stopping_rounds=EARLY_STOPPING_ROUNDS):
        #引数を反映
        cv_params['random_state'] = [seed]
        self.tuning_params=cv_params
        self.seed = seed
        self.cv_num = cv_num
        self.early_stopping_rounds = early_stopping_rounds
        start = time.time()
        #XGBoostのインスタンス作成
        cv_model = xgb.XGBRegressor()
        # グリッドサーチのインスタンス作成
        # n_jobs=-1にするとCPU100%で全コア並列計算。とても速い。
        cv = GridSearchCV(cv_model, cv_params, cv=cv_num, scoring=self.SCORING, n_jobs =-1)
        
        #グリッドサーチ実行
        evallist = [(self.X, self.y)]
        cv.fit(self.X,
                self.y,
                eval_set=evallist,
                early_stopping_rounds=early_stopping_rounds
                )
        self.elapsed_time = time.time() - start

        #最適パラメータの表示
        print('最適パラメータ ' + str(cv.best_params_))
        print('変数重要度' + str(cv.best_estimator_.feature_importances_))

        #特徴量重要度の取得と描画
        self.feature_importances = cv.best_estimator_.feature_importances_
        features = list(reversed(self.X_colnames))
        importances = list(reversed(cv.best_estimator_.feature_importances_.tolist()))
        plt.barh(features,importances)

        #グリッドサーチでの探索結果を返す
        return cv.best_params_

    #ランダムサーチ＋クロスバリデーション
    def random_search_tuning(self, cv_params=CV_PARAMS_RANDOM, cv_num=CV_NUM, n_iter=N_ITER_RANDOM, seed=SEED, early_stopping_rounds=EARLY_STOPPING_ROUNDS):
        #引数を反映
        cv_params['random_state'] = [seed]
        self.tuning_params=cv_params
        self.seed = seed
        self.cv_num = cv_num
        self.early_stopping_rounds = early_stopping_rounds
        start = time.time()  # 処理時間測定
        #XGBoostのインスタンス作成
        cv_model = xgb.XGBRegressor()
        # ランダムサーチのインスタンス作成
        # n_jobs=-1にするとCPU100%で全コア並列計算。とても速い。
        cv = RandomizedSearchCV(cv_model, cv_params, cv = cv_num, random_state=seed, n_iter=n_iter, scoring=self.SCORING, n_jobs =-1)
        
        #グリッドサーチ実行
        evallist = [(self.X, self.y)]
        cv.fit(self.X,
                self.y,
                eval_set=evallist,
                early_stopping_rounds=early_stopping_rounds
                )
        self.elapsed_time = time.time() - start

        #最適パラメータの表示
        print('最適パラメータ ' + str(cv.best_params_))
        print('変数重要度' + str(cv.best_estimator_.feature_importances_))

        #特徴量重要度の取得と描画
        self.feature_importances = cv.best_estimator_.feature_importances_
        features = list(reversed(self.X_colnames))
        importances = list(reversed(cv.best_estimator_.feature_importances_.tolist()))
        plt.barh(features,importances)

        #グリッドサーチでの探索結果を返す
        return cv.best_params_

    #ベイズ最適化時の評価指標算出メソッド(bayes_optは指標を最大化するので、RMSE等のLower is betterな指標は符号を負にして返す)
    def xgb_reg_evaluate(self, learning_rate, min_child_weight, subsample, colsample_bytree, max_depth):
        # 最適化対象のパラメータ
        params = {'learning_rate':learning_rate,              
                'min_child_weight': int(min_child_weight),
                'max_depth': int(max_depth),
                'colsample_bytree': colsample_bytree,
                'subsample': subsample,
                }
        params.update(self.BAYES_NOT_OPT_PARAMS)  # 最適化対象以外のパラメータも追加
        params['random_state'] = self.seed
        # XGBoostのモデル作成
        cv_model = xgb.XGBRegressor()
        cv_model.set_params(**params)
        # クロスバリデーションを実施
        kf = KFold(n_splits=self.cv_num, shuffle=True, random_state=self.seed)

        #cross_val_scoreでクロスバリデーション
        fit_params={'early_stopping_rounds':self.early_stopping_rounds, "eval_set" : [(self.X, self.y)], 'verbose': 0}
        scores = cross_val_score(cv_model, self.X, self.y, cv = kf, scoring=self.SCORING, fit_params = fit_params, n_jobs =-1)
        val = scores.mean()

        #スクラッチでクロスバリデーション
        # scores = []
        # for train, test in kf.split(self.X, self.y):
        #     X_train = self.X[train]
        #     y_train = self.y[train]
        #     X_test = self.X[test]
        #     y_test = self.y[test]
        #     cv_model.fit(X_train,
        #              y_train,
        #              eval_set=[(X_train, y_train)],
        #              early_stopping_rounds=self.early_stopping_rounds,
        #              verbose=0
        #              )
        #     pred = cv_model.predict(X_test)
        #     score = r2_score(y_test, pred)
        #     scores.append(score)
        # val = sum(scores)/len(scores)

        return val

    #ベイズ最適化(bayes_opt)
    def bayes_opt_tuning(self, beyes_params=BAYES_PARAMS, cv_num=CV_NUM, n_iter=N_ITER_BAYES, init_points=INIT_POINTS, acq=ACQ, seed=SEED, early_stopping_rounds=EARLY_STOPPING_ROUNDS):
        # 引数を反映
        self.tuning_params=beyes_params
        self.seed = seed
        self.cv_num = cv_num
        self.early_stopping_rounds = early_stopping_rounds
        # ベイズ最適化を実行
        start = time.time()
        xgb_bo = BayesianOptimization(self.xgb_reg_evaluate, beyes_params, random_state=seed)
        xgb_bo.maximize(init_points=15, n_iter=n_iter, acq=acq)
        self.elapsed_time = time.time() - start
        # 評価指標が最大となったときのパラメータを取得
        best_params = xgb_bo.max['params']
        best_params['min_child_weight'] = int(best_params['min_child_weight'])  # 小数で最適化されるのでint型に直す
        best_params['max_depth'] = int(best_params['max_depth'])  # 小数で最適化されるのでint型に直す
        # 最適化対象以外のパラメータも追加
        best_params.update(self.BAYES_NOT_OPT_PARAMS)
        best_params['random_state'] = self.seed
        # 特徴量重要度算出のため学習
        model = xgb.XGBRegressor()
        model.set_params(**best_params)
        evallist = [(self.X, self.y)]
        model.fit(self.X,
                self.y,
                eval_set=evallist,
                early_stopping_rounds=self.early_stopping_rounds
                )
        self.feature_importances = model.feature_importances_
        return best_params

    # 性能評価(leave_one_out)