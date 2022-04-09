import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import abc

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, r2_score
from .preprocess import split_data
from .configs import *
from .utils import auc_for_reg


class AbstractModel(abc.ABC):
    def __init__(self, pipeline, cross_val_args, df_train_data_all, df_train_label_all, label, **kwargs):
        self.pipeline = pipeline
        self.cross_validator = GridSearchCV(self.pipeline, **cross_val_args)
        test_size = kwargs.get("test_size", conf_data_split_args["test_size"])
        random_state = kwargs.get("random_state", conf_data_split_args["random_state"])
        if isinstance(df_train_data_all, np.ndarray) and isinstance(df_train_label_all, np.ndarray):
            self.X, self.y = df_train_data_all, df_train_label_all
        else:
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
    def predict_and_evaluate(self, **kwargs):
        raise NotImplementedError


class BinaryClassifier(AbstractModel):
    def __init__(self, pipeline, cross_val_args, df_train_data_all, df_train_label_all, label, **kwargs):
        super(BinaryClassifier, self).__init__(pipeline, cross_val_args, df_train_data_all, df_train_label_all,
                                               label, **kwargs)

    def predict_and_evaluate(self, if_plot=False):
        prob_pred = self.best_model.predict_proba(self.X_test)  # (N, 2)
        positive_probs = prob_pred[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, positive_probs)
        auc_score = auc(fpr, tpr)
        if if_plot:
            fig, axis = plt.subplots()
            axis.plot(fpr, tpr)
            axis.grid(True)
            axis.set_title(f"AUC: {auc_score:.3f}")
            plt.show()
            plt.close()

        return positive_probs, auc_score


class SVRClassifier(AbstractModel):
    def __init__(self, pipeline, cross_val_args, df_train_data_all, df_train_label_all, label, **kwargs):
        super(SVRClassifier, self).__init__(pipeline, cross_val_args, df_train_data_all, df_train_label_all,
                                               label, **kwargs)

    def predict_and_evaluate(self, **kwargs):
        pred = self.best_model.predict(self.X_test)
        auc_score = auc_for_reg(self.y_test, pred)

        return pred, auc_score


class Regressor(AbstractModel):
    def __init__(self, pipeline, cross_val_args, df_train_data_all, df_train_label_all, label, **kwargs):
        super(Regressor, self).__init__(pipeline, cross_val_args, df_train_data_all, df_train_label_all,
                                               label, **kwargs)

    def predict_and_evaluate(self, **kwargs):
        pred = self.best_model.predict(self.X_test)
        r2_score_val = r2_score(self.y_test, pred)

        return pred, r2_score_val


class SequenceModel(nn.Module):
    def __init__(self, params):
        """
        params: in the form of configs.conf_seq_model_params
        """
        super(SequenceModel, self).__init__()
        self.params = params
        self.embed_dim = self.params["transformer_encoder_layer_params"]["d_model"]
        self.head_dim = self.embed_dim // self.params["transformer_encoder_layer_params"]["nhead"]
        self.embed = nn.Linear(self.params["input_dim"], self.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(**self.params["transformer_encoder_layer_params"])
        self.encoder = nn.TransformerEncoder(encoder_layer, **self.params["transformer_encoder_params"])
        # self.head = nn.Sequential(
        #     nn.Linear(self.embed_dim + 1, len(TESTS) + len(VITALS)),  # +1: append "age",
        #     nn.ReLU(),
        #     nn.Linear(len(TESTS) + len(VITALS), len(TESTS) + len(VITALS))
        # )
        self.head = nn.Linear(self.embed_dim, len(TESTS) + len(VITALS))

    def forward(self, X_age, X_features, mask=None):
        # X_age: (B,), X_features: (B, T, N_features)
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(X_features.shape[1]).to(conf_device)
        X_emb = self.embed(X_features)  # (B, T, N_emb)
        X_encoded = self.encoder(X_emb.permute(1, 0, 2), mask)  # (B, T, N_emb) -> (T, B, N_emb) -> (T, B, N_emb)
        # X = torch.cat([X_encoded[-1, ...], X_age.view(-1, 1)], dim=1)  # (B, N_emb + 1)
        X = X_encoded[-1, ...]
        X = self.head(X)  # (B, N_cls + N_reg)
        assert len(TESTS) + len(VITALS) == X.shape[1]
        X_cls, X_reg = torch.sigmoid(X[:, :len(TESTS)]), X[:, len(TESTS):]

        return X_cls, X_reg
