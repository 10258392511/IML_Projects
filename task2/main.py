import pandas as pd
import helpers.configs as configs

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from helpers.preprocess import min_max_features

if __name__ == '__main__':
    # subtask 2
    label_subtask2 = "LABEL_Sepsis"
    train_path = "data/train_features.csv"
    train_label_path = "data/train_labels.csv"
    test_path = "data/test_features.csv"

    df_train_orig = pd.read_csv(train_path)
    df_train_labels = pd.read_csv(train_label_path)
    df_test_orig = pd.read_csv(test_path)

    df_train, df_train_labels, df_test = min_max_features(df_train_orig, df_train_labels, df_test_orig, "Age")

    pipeline = Pipeline(steps=[
        ("normalizer", configs.conf_normalizer_args["class"](**configs.conf_normalizer_args["args"])),
        ("svc", SVC(**configs.conf_SVC_args))
    ])
    # from grid search
    best_params = {
        "svc__C": 0.001,
        "svc__gamma": 0.011,
        "svc__class_weight": {0: 1, 1: 9}
    }
    pipeline.set_params(**best_params)
    print("training...")
    pipeline.fit(df_train.values, df_train_labels[label_subtask2].values)

    test_pids = df_test_orig["pid"].sort_values().unique()
    y_pred = pipeline.predict_proba(df_test.values)
    y_pred = y_pred[:, 1]
    out_df = pd.DataFrame(data={"pid": test_pids, label_subtask2: y_pred})
    # print(out_df[out_df[label_subtask2] > 0.5])
    out_df.to_csv("data/subtask2.csv", float_format="%.3f", index=False)
