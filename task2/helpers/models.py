import numpy as np
import pandas as pd
import abc

from sklearn.model_selection import GridSearchCV
from .preprocess import split_data
from .configs import *


class AbstractModel(abc.ABC):
    def __init__(self, pipeline, cross_val_args, df_train_data_all, df_train_label_all, label, **kwargs):
        self.pipeline = pipeline
        self.cross_validator = GridSearchCV(self.pipeline, **cross_val_args)
        test_size = kwargs.get("test_size", conf_data_split_args["test_size"])
        random_state = kwargs.get("random_state", conf_data_split_args["random_state"])
        self.X, self.y = df_train_data_all.values, df_train_label_all[label].values
        self.X_train, self.y_train, self.X_test, self.y_test = split_data(df_train_data_all, df_train_label_all, label,
                                                                          test_size=test_size, random_state=random_state)
        self.best_model = None
        self.best_params = None
        self.find_best_model()

    def find_best_model(self):
        self.cross_validator.fit(self.X_train, self.y_train)
        self.best_model = self.cross_validator.best_estimator_
        self.best_params = self.cross_validator.best_params_

    def refit_all_data(self):
        # assert self.best_params is not None
        self.pipeline.set_params(**self.best_params)
        self.pipeline.fit(self.X, self.y)

    @abc.abstractmethod
    def predict_and_evaluate(self):
        raise NotImplementedError


class BinaryClassifier(AbstractModel):
    def __init__(self, pipeline, cross_val_args, df_train_data_all, df_train_label_all, label, **kwargs):
        super(BinaryClassifier, self).__init__(pipeline, cross_val_args, df_train_data_all, df_train_label_all,
                                               label, **kwargs)

    def predict_and_evaluate(self):
        pass


class Regressor(AbstractModel):
    pass
