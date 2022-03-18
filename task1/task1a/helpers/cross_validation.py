import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV


class CrossValidator(object):
    def __init__(self, X, y, classifier, param_grid: dict, K, scoring: dict, **kwargs):
        self.X = X
        self.y = y
        self.clf = GridSearchCV(classifier, param_grid, cv=K, scoring=scoring, refit=kwargs.get("refit", False))
        self.cv_result = None

    def fit(self):
        self.clf.fit(self.X, self.y)
        self.cv_result = pd.DataFrame(self.clf.cv_results_)

        return self.cv_result

    def get_rmse(self):
        assert self.cv_result is not None

        # minus sign due to "greater_is_better" is False in utils. rmse_scoring(.)
        return -self.cv_result["mean_test_rmse_score"]
    
