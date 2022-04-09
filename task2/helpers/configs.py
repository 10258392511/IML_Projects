import torch

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from imblearn.over_sampling import SMOTE
from .utils import auc_for_reg, neg_rmse


conf_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
TESTS = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis']

conf_classification_columns = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos',
                          'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                          'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis']

# conf_imputer_args = {
#     "class": KNNImputer,
#     "args": {
#         "n_neighbors": 5,
#         "weights": "distance"
#     }
# }

conf_imputer_args = {
    "class": SimpleImputer,
    "args": {
        "strategy": "mean"
    }
}

conf_over_sampler_args = {
    "class": SMOTE,
    "args": {
        "sampling_strategy": "auto",
        "random_state": 0,
        "k_neighbors": 5
    }
}

conf_normalizer_args = {
    "class": StandardScaler,
    "args": {}
}

conf_SVC_args = {
    "probability": True
}

# conf_lin_SVC_args = {
#     "C": 1,
#     "class_weight": {0: 1, 1: 100},
#     "max_iter": 10000
# }

# conf_lin_SVC_val_args = {
#     "param_grid": {
#         "svc__C": [0.1, 1, 10],
#         "svc__class_weight": [{0: 1, 1: 500}, {0: 1, 1: 100}, {0: 1, 1: 10}]
#     },
#     "scoring": ["roc_auc", "accuracy", "f1", "precision", "recall"],
#     "refit": "roc_auc",
#     "cv": StratifiedKFold(n_splits=3, shuffle=True, random_state=11),
#     "verbose": 3
# }

conf_SVC_cross_val_args = {
    "param_grid": {
        "svc__C": [0.009, 0.01, 0.011],
        "svc__gamma": [0.04, 0.05, 0.06],
        "svc__kernel": ["rbf"],
        "svc__class_weight": [{0: 1, 1: 9}, {0: 1, 1: 10}, {0: 1, 1: 11}]
    },
    "scoring": ["roc_auc", "accuracy"],  # ["roc_auc", "accuracy", "f1", "precision", "recall"],
    "refit": "roc_auc",
    "cv": StratifiedKFold(n_splits=3, shuffle=True, random_state=11),
    "verbose": 3
}

conf_logistic_cross_val_args = {
    "param_grid": {
        "log__C": [0.1, 1],
    },
    "scoring": ["roc_auc", "accuracy", "f1", "precision", "recall"],
    "refit": "roc_auc",
    "cv": StratifiedKFold(n_splits=5, shuffle=True, random_state=11),
    "verbose": 3
}

conf_SVR_cls_cross_val_args = {
    "param_grid": {
        "svr__C": [10, 1],
        "svr__gamma": [1],
        "svr__kernel": ["rbf"]
    },
    "scoring": {
        "auc": make_scorer(auc_for_reg),
        "neg_root_mean_squared_error": make_scorer(neg_rmse)
    },
    "refit": "auc",
    "cv": 5,
    "verbose": 3
}

# conf_SVR_cls_cross_val_args = {
#     "param_grid": {
#         "sgd__alpha": [0.0001, 0.001],
#         "sgd__l1_ratio": [0.15, 0.5]
#     },
#     "scoring": {
#         "auc": make_scorer(auc_for_reg),
#         "neg_root_mean_squared_error": make_scorer(neg_rmse)
#     },
#     "refit": "auc",
#     "cv": 5,
#     "verbose": 3
# }

conf_SVR_cross_val_args = {
    "param_grid": {
        "svr__C": [1, 10, 100],
        "svr__gamma": [1],
        "svr__kernel": ["rbf"]
    },
    "scoring": ["r2", "neg_root_mean_squared_error"],
    "refit": "r2",
    "cv": 3,
    "verbose": 3
}

conf_data_split_args = {
    "test_size": 0.1,
    "random_state": 0
}


conf_seq_model_params = {
    "input_dim": 34,
    "transformer_encoder_layer_params": {
        "d_model": 64,
        "nhead": 8,
        "dim_feedforward": 64,
    },
    "transformer_encoder_params": {
        "num_layers": 3
    }
}

conf_auc_params = {
    "auc_p": 2,
    "auc_gamma": 0.7
}

conf_seq_model_opt_params = {
    "constructor": torch.optim.AdamW,
    "opt_params": {
        "lr": 2e-3
    }
}
