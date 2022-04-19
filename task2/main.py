import pandas as pd
import helpers.configs as configs

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from helpers.preprocess import min_max_features


def subtask1():
    # TODO: fill in training procedure; returns pd.DataFrame
    result_path = "data/subtask1(1).csv"
    df_result = pd.read_csv(result_path)

    return df_result


def subtask2():
    label_subtask2 = "LABEL_Sepsis"
    train_path = "data/train_features.csv"
    train_label_path = "data/train_labels.csv"
    test_path = "data/test_features.csv"

    df_train_orig = pd.read_csv(train_path)
    df_train_labels = pd.read_csv(train_label_path)
    df_test_orig = pd.read_csv(test_path)

    df_train, df_train_labels, df_test = min_max_features(df_train_orig, df_train_labels, df_test_orig, "Age")

    pipeline = Pipeline(steps=[
        ("norm", configs.conf_normalizer_args["class"](**configs.conf_normalizer_args["args"])),
        ("svc", SVC(**configs.conf_SVC_args))
    ])

    # from grid search
    best_params = {
        "svc__C": 0.001,
        "svc__gamma": 0.011,
        "svc__class_weight": {0: 1, 1: 9}
    }

    pipeline.set_params(**best_params)
    print("subtask2 training...")
    pipeline.fit(df_train.values, df_train_labels[label_subtask2].values)

    test_pids = df_test_orig["pid"].sort_values().unique()
    y_pred = pipeline.predict_proba(df_test.values)
    y_pred = y_pred[:, 1]
    data_out = {"pid": test_pids, label_subtask2: y_pred}
    df_out = pd.DataFrame(data=data_out)

    return df_out


def subtask3():
    labels_subtask3 = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
    train_path = "data/train_features.csv"
    train_label_path = "data/train_labels.csv"
    test_path = "data/test_features.csv"

    df_train_orig = pd.read_csv(train_path)
    df_train_labels = pd.read_csv(train_label_path)
    df_test_orig = pd.read_csv(test_path)

    df_train, df_train_labels, df_test = min_max_features(df_train_orig, df_train_labels, df_test_orig, "Age")

    # from grid search
    best_params = [
        {
            "svr__C": 300,
            "svr__gamma": 0.0001,
            "svr__kernel": "rbf"
         },
        {
            "svr__C": 500,
            "svr__gamma": 0.0001,
            "svr__kernel": "rbf"
        },
        {
            "svr__C": 500,
            "svr__gamma": 0.0001,
            "svr__kernel": "rbf"
        },
        {
            "svr__C": 500,
            "svr__gamma": 0.0001,
            "svr__kernel": "rbf"
        }
    ]

    test_pids = df_test_orig["pid"].sort_values().unique()
    svr_pipeline = Pipeline(steps=[
        ("norm", configs.conf_normalizer_args["class"](**configs.conf_normalizer_args["args"])),
        ("svr", SVR())
    ])

    data_out = {"pid": test_pids}
    for label, best_param in zip(labels_subtask3, best_params):
        svr_pipeline.set_params(**best_param)
        print(f"subtask3 training: {label}...")
        svr_pipeline.fit(df_train.values, df_train_labels[label].values)
        y_pred = svr_pipeline.predict(df_test.values)
        data_out[label] = y_pred

    df_out = pd.DataFrame(data=data_out)

    return df_out


if __name__ == '__main__':
    df_subtask1 = subtask1()
    df_subtask2 = subtask2()
    df_subtask3 = subtask3()
    df_out = pd.concat([df_subtask1, df_subtask2.drop("pid", axis=1), df_subtask3.drop("pid", axis=1)], axis=1)
    out_path = "data/task2_results.zip"
    df_compression_args = dict(method="zip", archive_name="task2_results.csv")
    df_out.to_csv(out_path, float_format="%.3f", index=False, compression=df_compression_args)

