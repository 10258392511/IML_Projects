import os
import random
import numpy as np
import torch
import zipfile

from urllib.request import urlretrieve
from pprint import pprint

if __name__ == '__main__':
    from configs import *
else:
    from .configs import *


def convert_txt_to_paths(filename: str):
    """
    Layout of data/:
    + food
    - sample.txt
    - test_triplets.txt
    -train_triplets.txt

    Returns: [(.jpg, .jpg, .jpg)...]
    """
    assert filename.find(".txt") >= 0, "invalid file type"
    filename = os.path.abspath(filename)
    parent_dir = os.path.dirname(filename)
    data_dir = os.path.join(parent_dir, "food")
    paths = []
    with open(filename, "r") as rf:
        line = rf.readline().strip()
        while len(line) > 0:
            paths.append(tuple([os.path.join(data_dir, f"{number}.jpg") for number in line.split(" ")]))
            line = rf.readline().strip()

    return paths


def train_test_split(paths: list, train_test_split=None, seed=0):
    if train_test_split is None:
        train_test_split = configs_train_test_split_ratio
    random.seed(seed)
    random.shuffle(paths)
    split_index = int(len(paths) * train_test_split)
    train_paths = paths[:split_index]
    test_paths = paths[split_index:]

    return train_paths, test_paths


def create_param_save_path(param_save_dir, filename):
    if not os.path.isdir(param_save_dir):
        os.makedirs(param_save_dir)

    return os.path.join(param_save_dir, filename)


def save_results(result: np.ndarray, save_path: str):
    assert save_path.find(".txt") >= 0, "invalid file type"
    result = result.reshape((-1, 1))
    np.savetxt(save_path, result, fmt="%1d")


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


if __name__ == '__main__':
    # # test convert_txt_to_paths(.)
    # for filename in ["../data/train_triplets.txt", "../data/test_triplets.txt"]:
    #     paths = convert_txt_to_paths(filename)
    #     pprint(paths[:5])
    #     print(f"number samples: {len(paths)}")
    #     print("-" * 50)

    # # test train_test_split(.)
    # train_all_filename = "../data/train_triplets.txt"
    # train_paths_all = convert_txt_to_paths(train_all_filename)
    # train_paths, test_paths = train_test_split(train_paths_all)
    # pprint(train_paths[:5])
    # print("-" * 50)
    # pprint(test_paths[:5])
    # print("-" * 50)
    # print(f"{len(train_paths)}, {len(test_paths)}")

    # # test save_results(.)
    # B = 100
    # X = np.random.randint(0, 2, size=(B,))
    # save_filename = "../exps/results/test_save_results.txt"
    # save_results(X, save_filename)

    # # test downloading file and unzipping it
    # url = "https://polybox.ethz.ch/index.php/s/39L5nDkzCNEhJ6J/download"
    # save_dir = "../data_download"
    # save_filename = "food.zip"
    # print("downloading...")
    # download_file(url, save_dir, save_filename)
    # save_path = os.path.join(save_dir, save_filename)
    # print("unzipping...")
    # unzip_file(save_path, save_dir)
    pass
