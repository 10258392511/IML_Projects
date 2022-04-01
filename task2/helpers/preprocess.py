import numpy as np
import pandas as pd

from pprint import pprint
from configs import *


def summarize_dataset(df_train, df_train_label):
    """
    Summarizes:
    (1). Portion of NaN's in each column
    (2). Portion of positive label in each classification task
    """
    df_train_nans = df_train.isna().sum() / df_train.shape[0]
    df_train_label_pos = df_train_label[conf_classification_columns].sum() / df_train.shape[0]
    print("Portion of NaN's: ")
    pprint(df_train_nans)
    print("-" * 50)
    print("Portion of positives: ")
    pprint(df_train_label_pos)


def aggregate_by_mean_and_impute(df_train, if_save=True, save_path=None):
    """
    We may change to better strategy to impute the data.
    """
    if if_save:
        assert save_path is not None
    df_train_agg_mean = df_train.groupby("pid").agg("mean")  # NaN's automatically skipped
    imputer = conf_imputer_args["class"](**conf_imputer_args["args"])
    array_train_imputed = imputer.fit_transform(df_train_agg_mean)
    df_train_imputed = pd.DataFrame(data=array_train_imputed, columns=df_train_agg_mean.columns)

    if if_save:
        df_train_imputed.to_csv(save_path)


if __name__ == '__main__':
    train_filename = "../data/train_features.csv"
    train_label_filename = "../data/train_labels.csv"
    df_train = pd.read_csv(train_filename)
    df_train_label = pd.read_csv(train_label_filename)
    summarize_dataset(df_train, df_train_label)

    print("Imputing data...")
    save_path = "../data/train_features_agg_imputed.csv"
    aggregate_by_mean_and_impute(df_train, if_save=True, save_path=save_path)
