import pandas as pd
import helpers.configs as configs

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from helpers.preprocess import aggregate_by_mean_and_impute

if __name__ == '__main__':
    # subtask 2
    label_subtask2 = "LABEL_Sepsis"
    train_filename = "data/train_features.csv"
    train_label_filename = "data/train_labels.csv"
    df_train_orig = pd.read_csv(train_filename)
    df_train = aggregate_by_mean_and_impute(df_train_orig, if_save=False)
    df_train_label = pd.read_csv(train_label_filename).sort_values("pid")
    df_train_features = df_train.loc[:, "Age":]

    pipeline = Pipeline(steps=[
        ("normalizer", configs.conf_normalizer_args["class"](**configs.conf_normalizer_args["args"])),
        ("svc", SVC(**configs.conf_SVC_args))
    ])
    # from grid search
    best_params = {
        "svc__C": 0.009,
        "svc__gamma": 0.006,
        "svc__class_weight": {0: 1, 1: 10}
    }
    pipeline.set_params(**best_params)
    print("training...")
    pipeline.fit(df_train_features.values, df_train_label[label_subtask2].values)

    test_filename = "data/test_features.csv"
    df_test_orig = pd.read_csv(test_filename)
    test_pids = df_test_orig["pid"].sort_values().unique()
    # fillna for df_test with df_train_mean
    df_test = df_test_orig.groupby("pid").agg("mean")
    df_train_mean = df_train.mean()
    for column in df_test.columns:
        if column not in ["Time", "Age"]:
            df_test[column].fillna(df_train_mean[column], inplace=True)
    # print(df_test.head())
    df_test_features = df_test.loc[:, "Age":]
    y_pred = pipeline.predict_proba(df_test_features.values)
    y_pred = y_pred[:, 1]
    out_df = pd.DataFrame(data={"pid": test_pids, label_subtask2: y_pred})
    # print(out_df[out_df[label_subtask2] > 0.5])
    out_df.to_csv("data/subtask2.csv", float_format="%.3f", index=False)
