import numpy as np
import pandas as pd
import pickle

from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pprint import pprint
from .configs import *


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
    # print(df_train_agg_mean.isna().sum() / df_train_agg_mean.shape[0])
    imputer = conf_imputer_args["class"](**conf_imputer_args["args"])
    array_train_imputed = imputer.fit_transform(df_train_agg_mean)
    df_train_imputed = pd.DataFrame(data=array_train_imputed, columns=df_train_agg_mean.columns)

    if if_save:
        df_train_imputed.to_csv(save_path)

    return df_train_imputed


def split_data(df_train: pd.DataFrame, df_train_label: pd.DataFrame, label, test_size=0.1, random_state=0):
    if isinstance(df_train, np.ndarray) and isinstance(df_train_label, np.ndarray):
        X, y = df_train, df_train_label
    else:
        X, y = df_train.values, df_train_label[label].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, y_train, X_test, y_test


def preprocess_for_transformer(feature_path, label_path=None):
    df = pd.read_csv(feature_path)
    df.fillna(0, inplace=True)
    pids = np.unique(df["pid"])
    if label_path is not None:
        df_label = pd.read_csv(label_path)
    out_dict = {}
    pbar = tqdm(pids, total=len(pids), desc="preprocessing...")
    for pid in pbar:
        df_selected = df[df["pid"] == pid]
        cls_labels, reg_labels = None, None
        if label_path is not None:
            df_label_selected = df_label[df_label["pid"] == pid]
            cls_labels = df_label_selected.loc[:, TESTS].values
            reg_labels = df_label_selected.loc[:, VITALS].values
        out_dict[pid] = {
            "age": df_selected["Age"].values[0],
            "data": df_selected.loc[:, "EtCO2":].values,
            "cls_labels": cls_labels,
            "reg_labels": reg_labels
        }

    return pids, out_dict


class Task2Dataset(Dataset):
    def __init__(self, data_filename, data=None):
        super(Task2Dataset, self).__init__()
        if data is None:
            with open(data_filename, "rb") as rf:
                self.data = pickle.load(rf)
        else:
            self.data = data

    def __len__(self):
        return len(self.data["pids"])

    def __getitem__(self, index):
        pid = self.data["pids"][index]
        data = self.data["data"][pid]
        if data["cls_labels"] is None:
            data["cls_labels"] = 0
        if data["reg_labels"] is None:
            data["reg_labels"] = 0
        return data["age"], data["data"], data["cls_labels"], data["reg_labels"]
