import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from .utils import rmse
from .config import config_split_ratio


class CrossValidator(object):
    def __init__(self, X, y, classifier, param_grid: dict, K, scoring: dict, seed=0, **kwargs):
        split_ratio = kwargs.get("split_ratio", config_split_ratio)
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(X, y, train_size=int(X.shape[0] * split_ratio),
                                                                      random_state=seed)
        self.clf = GridSearchCV(classifier, param_grid, cv=K, scoring=scoring, refit=kwargs.get("refit", False))
        self.cv_result = None

    def fit(self):
        self.clf.fit(self.X_tr, self.y_tr)
        self.cv_result = pd.DataFrame(self.clf.cv_results_)

        return self.cv_result

    # def get_rmse(self):
    #     assert self.cv_result is not None
    #
    #     # minus sign due to "greater_is_better" is False in utils. rmse_scoring(.)
    #     return -self.cv_result["mean_test_rmse_score"]

    def get_coeff(self):
        assert self.cv_result is not None

        return self.clf.best_estimator_.coef_

    def predict(self):
        assert self.cv_result is not None
        y_pred = self.clf.best_estimator_.predict(self.X_te)
        rmse_val = rmse(self.y_te, y_pred)

        return y_pred, rmse_val
