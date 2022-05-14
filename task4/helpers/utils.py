import numpy as np
import pandas as pd
import os
import pickle
import zipfile

from typing import List
from urllib.request import urlretrieve
from collections import OrderedDict
from .configs import config_split_ratio


def read_data(filename_feat: str, filename_label: str = None):
    assert ".csv" in filename_feat, "must be .csv files"
    if filename_label is not None:
        assert ".csv" in filename_label, "must be .csv files"

    df_feat = pd.read_csv(filename_feat)
    ids = df_feat["Id"].values  # (N,), int64
    smiles = df_feat["smiles"].values  # (N,), str
    features = df_feat.loc[:, "feature_0000":].values.astype(int)  # (N, N_feat), int32, {0, 1}

    tgts = None
    if filename_label is not None:
        df_label = pd.read_csv(filename_label)
        dict_temp = {}
        for id_iter, tgt_iter in zip(df_label["Id"].values, df_label.iloc[:, 1]):
            dict_temp[id_iter] = tgt_iter

        tgts = np.array([dict_temp[id_iter] for id_iter in ids])  # (N,)

    # (N,), (N,), (N, N_feat), (N,) or None
    return ids, smiles, features, tgts


def create_vocabulary(all_filenames: List[str], if_save=True, save_path="../info/vocab.pkl"):
    # only need features
    counter = OrderedDict()
    for filename in all_filenames:
        print(f"current: {filename}")
        _, smiles, _, _ = read_data(filename)
        for smile_iter in smiles:
            for c_iter in smile_iter:
                if c_iter not in counter:
                    counter[c_iter] = 0
                else:
                    counter[c_iter] += 1

    i2c = list(counter.keys()) + ["<BOS>", "<EOS>", "<PAD>"]
    c2i = {smile_iter: ind for ind, smile_iter in enumerate(i2c)}

    if if_save:
        assert save_path is not None
        with open(save_path, "wb") as wf:
            pickle.dump({"counter": counter, "i2c": i2c, "c2i": c2i}, wf)

    return counter, i2c, c2i


def split_train_set(filename_feat, filename_label, seed=None, mode="train"):
    assert mode in ("train", "eval")
    ids, smiles, features, tgts = read_data(filename_feat, filename_label)
    split_ind = int(ids.shape[0] * config_split_ratio)
    np.random.seed(seed)
    inds = np.arange(ids.shape[0])
    np.random.shuffle(inds)
    if mode == "train":
        sel_inds = inds[:split_ind]
    else:
        sel_inds = inds[split_ind:]

    return ids[sel_inds], smiles[sel_inds], features[sel_inds], tgts[sel_inds]


def create_param_save_path(param_save_dir, filename):
    if not os.path.isdir(param_save_dir):
        os.makedirs(param_save_dir)

    return os.path.join(param_save_dir, filename)


def create_log_dir(time_stamp: str, arg_dict: dict):
    dir_name = time_stamp
    for key, val in arg_dict.items():
        if not isinstance(val, str):
            dir_name += f"_{key}_{val:.3f}"
        else:
            dir_name += f"_{key}_{val}"

    return dir_name.replace(".", "_")


def download_file(url, save_dir, save_filename):
    print("downloading...")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_filename)
    urlretrieve(url, save_path)
    print("Done!")


def unzip_file(filename, save_dir):
    print("unzipping...")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with zipfile.ZipFile(filename) as zip_rf:
        zip_rf.extractall(save_dir)
    print("Done!")
