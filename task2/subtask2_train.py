import time
import numpy as np
import pandas as pd
import helpers.configs as configs

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from helpers.preprocess import min_max_features
from helpers.models import BinaryClassifier

if __name__ == '__main__':
    train_path = "data/train_features.csv"
    train_label_path = "data/train_labels.csv"
    test_path = "data/test_features.csv"

    df_train_orig = pd.read_csv(train_path)
    df_train_labels = pd.read_csv(train_label_path)
    df_test_orig = pd.read_csv(test_path)

    df_train, df_train_labels, df_test = min_max_features(df_train_orig, df_train_labels, df_test_orig, "Age")

    svc_pipeline = Pipeline(steps=[
        ("norm", configs.conf_normalizer_args["class"](**configs.conf_normalizer_args["args"])),
        ("svc", SVC(**configs.conf_SVC_args))
    ])

    clf = BinaryClassifier(svc_pipeline, configs.conf_SVC_cross_val_args, df_train, df_train_labels, "LABEL_Sepsis")
    timestamp = f"{time.time()}".replace(".", "_")
    pd.DataFrame(clf.cross_validator.cv_results_).to_csv(f"data/clf_cv_results_{timestamp}.csv")
