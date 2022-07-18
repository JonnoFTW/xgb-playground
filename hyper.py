from hyperopt import hp, fmin, STATUS_OK, Trials, tpe
import xgboost as xgb


class HyperOptTuner:
    def __init__(self, train, validation, early_stopping=10, max_evals=100):
        self.count = 0
        self.train = train
        self.validation = validation

        self.early_stopping = early_stopping
        self.tuned_params = None
        self.max_evals = max_evals

    def _score(self, params):
        """
        The function we want to minimize, using the currently selected params
        :param params:
        :return:
        """
        self.count += 1

        # split self.train and into train_test for k-fold cross validation
        eval_list = [(self.train, 'train'), (self.validation, 'eval')]
        estimators = int(params['num_boost_round'])
        del params['num_boost_round']
        model = xgb.train(
            params,
            self.train,
            num_boost_round=estimators,
            early_stopping_rounds=self.early_stopping,
            evals=eval_list,
            verbose_eval=False,
        )

        score = model.best_score

        return {
            'loss': score,
            'status': STATUS_OK,
            'params': params,
            'model': model
        }

    def optimize(self, space, algo=tpe.suggest):
        trials = Trials()
        fmin(
            self._score,
            space=space,
            algo=algo,
            trials=trials,
            max_evals=self.max_evals,
        )
        return trials


"""
space could be:
  space = {
            'n_estimators': 2000,  # hp.quniform('n_estimators', 10, 1000, 10),
            'eta': hp.quniform('eta', 0.025, 0.3, 0.025),
            'max_depth': hp.choice('max_depth', np.arange(1, 9, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 10, dtype=int)),
            'subsample': hp.quniform('subsample', 0.3, 1, 0.05),
            'gamma': hp.quniform('gamma', 0.1, 20, 0.1),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.25),
            'eval_metric': 'map',
            'objective': 'rank:pairwise',
            'silent': 1
        }

"""