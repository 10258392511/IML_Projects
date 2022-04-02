from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE


conf_classification_columns = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos',
                          'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                          'LABEL_Bilirubin_direct', 'LABEL_EtCO2', 'LABEL_Sepsis']

conf_imputer_args = {
    "class": KNNImputer,
    "args": {
        "n_neighbors": 5,
        "weights": "distance"
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

conf_SVC_cross_val_args = {
    "param_grid": {
        "svc__C": [0.1, 1],
        "svc__gamma": [1],
        "svc__kernel": ["rbf"]
    },
    "scoring": ["roc_auc", "accuracy", "f1", "precision", "recall"],
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
    "cv": StratifiedKFold(n_splits=3, shuffle=True, random_state=11),
    "verbose": 3
}

conf_data_split_args = {
    "test_size": 0.1,
    "random_state": 0
}
