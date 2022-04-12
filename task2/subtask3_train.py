import time
import numpy as np
import pandas as pd
import helpers.configs as configs

from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from helpers.preprocess import min_max_features
from helpers.models import Regressor

if __name__ == '__main__':
    train_path = "data/train_features.csv"
    train_label_path = "data/train_labels.csv"
    test_path = "data/test_features.csv"

    df_train_orig = pd.read_csv(train_path)
    df_train_labels = pd.read_csv(train_label_path)
    df_test_orig = pd.read_csv(test_path)

    df_train, df_train_labels, df_test = min_max_features(df_train_orig, df_train_labels, df_test_orig, "Age")

    svr_pipeline = Pipeline(steps=[
        ("norm", configs.conf_normalizer_args["class"](**configs.conf_normalizer_args["args"])),
        ("svr", SVR())
    ])

    # ridge_pipeline = Pipeline(steps=[
    #     ("norm", configs.conf_normalizer_args["class"](**configs.conf_normalizer_args["args"])),
    #     ("ridge", Ridge())
    # ])

    label = "LABEL_RRate"
    reg = Regressor(svr_pipeline, configs.conf_SVR_cross_val_args, df_train, df_train_labels, label)
    timestamp = f"{time.time()}".replace(".", "_")
    pd.DataFrame(reg.cross_validator.cv_results_).to_csv(f"data/reg_cv_results_{label}_{timestamp}.csv")
