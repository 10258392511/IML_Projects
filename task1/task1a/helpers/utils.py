import numpy as np
import pandas as pd
import typing

from sklearn.metrics import mean_squared_error, make_scorer


def read_data(filename, mode="train") -> typing.Tuple[np.ndarray, np.ndarray]:
    assert mode in ["train", "test"]
    if mode == "train":
        df = pd.read_csv(filename)
        y = df["y"].values
        X = df.iloc[:, 1:].values

    else:
        raise NotImplementedError("Not required for this project.")

    return X, y


def save_data(filename, data):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, header=False)


def rmse(y_true,  y_pred):
    mse = mean_squared_error(y_true, y_pred)

    return np.sqrt(mse)


def rmse_scoring():
    return make_scorer(rmse, greater_is_better=False)
