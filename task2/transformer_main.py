import numpy as np
import pandas as pd
import torch
import helpers.configs as configs

from torch.utils.data import DataLoader
from helpers.preprocess import preprocess_for_transformer, Task2Dataset
from helpers.models import SequenceModel


if __name__ == '__main__':
    test_features_path = "data/test_features.csv"
    model_path = "params/1649485431_4342546/model.pt"
    df_save_path = "data/test_preds.zip"
    df_compression_args = dict(method="zip", archive_name="test_preds.csv")

    pids, data = preprocess_for_transformer(test_features_path, None)
    test_dataset = Task2Dataset(data_filename=None, data={"pids": pids, "data": data})
    test_loader = DataLoader(test_dataset, batch_size=1024)
    model = SequenceModel(configs.conf_seq_model_params).to(configs.conf_device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("Predicting...")
    preds = []
    for X_ages, X_features, _, _ in test_loader:
        X_ages = X_ages.float().to(configs.conf_device)
        X_features = X_features.float().to(configs.conf_device)
        y_cls_pred, y_reg_pred = model(X_ages, X_features, None)
        preds.append(torch.cat([y_cls_pred, y_reg_pred], dim=1).detach().cpu().numpy())

    # preds: list[(B, N_preds)...]
    preds =np.concatenate(preds, axis=0)

    print("Saving prediction...")
    df = pd.DataFrame(data=preds, columns=configs.TESTS + configs.VITALS)
    df.insert(0, "pid", pids)
    # print(df.head())
    df.to_csv(df_save_path, index=False, float_format="%.3f", compression=df_compression_args)
