import os
import torch
import numpy as np
import pandas as pd
import helpers.configs as configs

from tqdm import tqdm
from torch.utils.data import DataLoader
from helpers.dataset import MoleculeDataset
from helpers.modules import EnergyEstimator
from helpers.utils import download_file


if __name__ == '__main__':
    # download dataset
    model_url = "https://polybox.ethz.ch/index.php/s/bBqFNmhuQR0pDpA/download"
    model_dir = "./model_params"
    model_filename = "model.pt"
    download_file(model_url, model_dir, model_filename)
    model_path = os.path.join(model_dir, model_filename)
    # model_path = "./params/train/2022_05_14_10_04_10_020088_batch_size_64_000_epochs_50_000/model.pt"


    params = {
        "batch_size": 128,
        "result_save_path": "./predictions.csv"
    }

    test_ds = MoleculeDataset("test", vocab_path="info/vocab.pkl")
    test_loader = DataLoader(test_ds, batch_size=params["batch_size"])
    energy_estimator_params = {
        "seq_encoder_params": configs.config_seq_encoder_params,
        "mlp_params": configs.config_mlp_params,
        "fusion_params": configs.config_fusion_params
    }

    energy_estimator = EnergyEstimator(energy_estimator_params).to(configs.config_device)
    energy_estimator.load_state_dict(torch.load(model_path))
    energy_estimator.eval()

    ids, preds = [], []
    pbar = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for X_ids, X_smiles, X_features, _ in pbar:
            ids.append(X_ids.numpy())
            X_smiles = X_smiles.to(configs.config_device).transpose(0, 1)
            X_features = X_features.to(configs.config_device)
            y_pred = energy_estimator(X_smiles, X_features)
            preds.append(y_pred.detach().cpu().numpy())

    ids, preds = np.concatenate(ids, axis=0), np.concatenate(preds, axis=0)
    df_test = pd.DataFrame(data={"Id": ids, "y": preds})
    df_test.to_csv(params["result_save_path"], index=False)
