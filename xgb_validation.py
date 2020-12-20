import xgboost as xgb
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
import pandas as pd
import numpy as np


class XGBRegressorValidation():

    # 共通定数
    SEED = 42  # デフォルト乱数シード
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # デフォルト複数乱数シード
    CV_NUM = 5  # クロスバリデーションの分割数
    EARLY_STOPPING_ROUNDS = 50  # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
    NUM_ROUNDS = 10000  # 学習時の最大学習回数(大きな値を指定すれば、実質的にはEARLY_STOPPING_ROUNDSに依存)
    #  評価指標一覧
    SCORES = {'r2': 'mean',
              'rmse': 'mean',
              'maxerror': 'max',
              'rmsle': 'mean'}

    def __init__(self, X, y, X_colnames, y_colname=None):
        """
        初期化

        Parameters
        ----------
        X : ndarray
            説明変数データ(pandasではなくndarray)
        y : ndarray
            目的変数データ
        X_colnames : list(str)
            説明変数のフィールド名
        y_colname : str
            目的変数のフィールド名
        """

        if X.shape[1] != len(X_colnames):
            raise Exception('width of X must be equal to length of X_colnames')
        self.X = X
        self.y = y
        self.X_colnames = X_colnames
        self.y_colname = y_colname

    def _train_and_predict(self, X_train, X_test, y_train, y_test, test_index, params, early_stopping_rounds):
        """
        学習データで学習し、テストデータで推論する

        Parameters
        ----------
        X_train : ndarray
            学習用説明変数データ
        X_test : ndarray
            テスト用説明変数データ
        y_train : ndarray
            学習用目的変数データ
        y_test : ndarray
            テスト用目的変数データ
        test_index : ndarray
            テスト用データのインデックス
        params : dict
            XGBoost使用パラメータ
        early_stopping_rounds : int
            学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)  # 学習用データ
        dtest = xgb.DMatrix(X_test, label=y_test)  # テストデータ
        evals = [(dtest, 'eval'), (dtrain, 'train')]  # 結果表示用の学習データとテストデータを指定
        evals_result = {}  # 結果保持用

        # 学習実行
        model = xgb.train(params,
                          dtrain,  # 訓練データ
                          num_boost_round=self.NUM_ROUNDS,
                          early_stopping_rounds=early_stopping_rounds,
                          evals=evals,
                          evals_result=evals_result
                          )
        # テストデータを推論
        pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
        detail_dict = {
            'data_index': test_index.tolist(),
            'pred_value': pred.tolist(),
            'real_value': y_test[:, 0].tolist(),
            'error': (y_test[:, 0] - pred).tolist(),
        }
        for i, colname in enumerate(self.X_colnames):
            detail_dict['X_' + colname] = X_test[:, i].tolist()
        
        return detail_dict

    def _calc_scores(self, real, pred, scores):
        """
        評価指標算出

        Parameters
        ----------
        real : list
            実際の値
        pred : list
            推論値
        scores : dict
            使用する評価指標(例:'r2', 'rmse', 'maxerror', 'rmsle')と集計方法(例:'mean', 'max')の辞書
        """
        score_dict = {}
        for score in scores.keys():
            if score == 'r2':
                score_dict['r2'] = r2_score(real, pred)
            elif score == 'rmse':
                score_dict['rmse'] = mean_squared_error(
                    real, pred, squared=False)
            elif score == 'rmsle':
                score_dict['rmsle'] = np.sqrt(mean_squared_log_error(
                    real, pred))
            elif score == 'maxerror':
                score_dict['maxerror'] = max(
                    [abs(p - r) for r, p in zip(real, pred)])
        return score_dict

    def cross_validation(self, params, scores=SCORES, cv=None, cv_num=CV_NUM, seed=SEED, early_stopping_rounds=EARLY_STOPPING_ROUNDS):
        """
        クロスバリデーション実行

        Parameters
        ----------
        params : dict
            XGBoost使用パラメータ
        scores : dict
            使用する評価指標(例:'r2', 'rmse', 'maxerror', 'rmsle')と集計方法(例:'mean', 'max')の辞書
        cv : KFold
            クロスバリデーション分割法(未指定時はcv_numに基づきkFoldで分割)
        cv_num : int
            分割法未指定時の、kFold分割数
        seed : int
            分割法未指定時の、kFold乱数シード
        early_stopping_rounds : int
            学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
        """
        # 分割法未指定時、cv_numとseedに基づきランダムに分割
        if cv is None:
            cv = KFold(n_splits=cv_num, shuffle=True, random_state=seed)

        # データの分割
        score_list = []
        detail_list = []
        cv_cnt = 0  # 交差検証の何回目かをカウント
        for train_index, test_index in cv.split(self.X, self.y):
            # 学習データとテストデータに分割
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            # 学習データで学習し、テストデータで推論
            detail_dict = self._train_and_predict(
                X_train, X_test, y_train, y_test, test_index, params, early_stopping_rounds=early_stopping_rounds)
            # 評価指標算出
            score_dict = self._calc_scores(detail_dict['real_value'], detail_dict['pred_value'], scores)
            # 結果をDataFrameに格納
            df_score = pd.DataFrame([score_dict])  # 指標
            df_score.insert(0, 'cv_cnt', cv_cnt)
            score_list.append(df_score)
            df_detail = pd.DataFrame(detail_dict)  # データ点ごとの結果詳細
            df_detail.insert(0, 'cv_cnt', cv_cnt)
            detail_list.append(df_detail)
            cv_cnt += 1

        # クロスバリデーションの全ての結果を結合
        score_all = pd.concat(score_list, ignore_index=True)
        detail_all = pd.concat(detail_list, ignore_index=True)
        # scoresの指定に合わせて集計
        validation_score = score_all.agg(scores)
        print(validation_score)
        # 詳細結果に指標を結合
        validation_detail = pd.merge(
            detail_all, score_all, on='cv_cnt', how='left')
        validation_detail = validation_detail.sort_values('data_index').reset_index()

        return validation_score, validation_detail

    # def cross_validation_group(self, )

    def leave_one_out(self, params, scores=SCORES, early_stopping_rounds=EARLY_STOPPING_ROUNDS):
        """
        Leave_One_Outクロスバリデーション実行

        Parameters
        ----------
        params : dict
            XGBoost使用パラメータ
        scores : dict
            使用する評価指標(例:'r2', 'rmse', 'maxerror', 'rmsle')と集計方法(例:'mean', 'max')の辞書
        early_stopping_rounds : int
            学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
        """
        loo = LeaveOneOut()
        # データの分割
        detail_list = []
        cv_cnt = 0  # 交差検証の何回目かをカウント
        for train_index, test_index in loo.split(X):
            # 学習データとテストデータに分割
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            # 学習データで学習し、テストデータで推論
            detail_dict = self._train_and_predict(
                X_train, X_test, y_train, y_test, test_index, params, early_stopping_rounds=early_stopping_rounds)
            # 結果をDataFrameに格納
            df_detail = pd.DataFrame(detail_dict)  # データ点ごとの結果詳細
            df_detail.insert(0, 'cv_cnt', cv_cnt)
            detail_list.append(df_detail)
            cv_cnt += 1
        # 全ての推論結果を結合
        detail_all = pd.concat(detail_list, ignore_index=True)
        # 評価指標算出(分割ごとに算出するcross_validationメソッドと異なり、全ての推論値と正解値から一括算出する)
        score_dict = self._calc_scores(detail_all['real_value'].values.tolist(), detail_all['pred_value'].values.tolist(), scores)
        validation_score = pd.DataFrame([score_dict])
        print(validation_score)
        # 詳細結果に指標を結合
        validation_detail = pd.merge(
            detail_all, score_all, on='cv_cnt', how='left')
        validation_detail = validation_detail.sort_values('data_index').reset_index()

        return validation_score, validation_detail

    # 乱数シードを変えてクロスバリデーション実行
