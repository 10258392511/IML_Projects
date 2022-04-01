from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


conf_classification_columns = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos',
                          'LABEL_Bilirubin_total', 'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
                          'LABEL_Bilirubin_direct', 'LABEL_EtCO2']


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
