import numpy as np
import pandas as pd
import typing

from sklearn.metrics import mean_squared_error, make_scorer


def read_data(filename, mode="train") -> typing.Tuple[np.ndarray, np.ndarray]:
    assert mode in ["train", "test"]
    if mode == "train":
        df = pd.read_csv(filename)
        y = df["y"].values
        X = df.loc[:, "x1":"x5"].values

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


def get_features(X):
    assert X.shape[1] == 5
    X_features = np.empty_like(X, shape=(X.shape[0], 21))
    X_features[:, -1] = 1
    X_features[:, :5] = X
    X_features[:, 5:10] = X ** 2
    X_features[:, 10:15] = np.exp(X)
    X_features[:, 15:20] = np.cos(X)

    # (N, 21)
    return X_features
