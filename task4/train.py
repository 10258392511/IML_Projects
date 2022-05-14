import argparse
import torch
import helpers.configs as configs

from datetime import datetime
from torch.utils.data import DataLoader
from helpers.dataset import MoleculeDataset
from helpers.modules import EnergyEstimator
from helpers.trainer import Trainer
from helpers.utils import create_log_dir


if __name__ == '__main__':
    """
    python train.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    params = vars(parser.parse_args())

    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    log_keys = ["batch_size", "epochs"]
    log_params = {key: params[key] for key in log_keys}
    log_dir = create_log_dir(time_stamp, log_params)
    params.update(configs.config_train_params)
    params.update({
        "log_dir": f"run/train/{log_dir}",
        "param_save_dir": f"params/train/{log_dir}",
        "if_notebook": False,
        "scheduler_params": {
            "gamma": 0.95
        }
    })

    # dataset
    train_ds = MoleculeDataset("train", seed=params["seed"], vocab_path="info/vocab.pkl")
    eval_ds = MoleculeDataset("eval", seed=params["seed"], vocab_path="info/vocab.pkl")
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=params["batch_size"])

    # model
    energy_estimator_params = {
        "seq_encoder_params": configs.config_seq_encoder_params,
        "mlp_params": configs.config_mlp_params,
        "fusion_params": configs.config_fusion_params
    }

    energy_estimator = EnergyEstimator(energy_estimator_params).to(configs.config_device)
    model_path = "./params/pre_train/2022_05_14_09_30_46_206853_batch_size_128_000_epochs_5_000/model.pt"
    energy_estimator.load_state_dict(torch.load(model_path))
    energy_estimator.eval()

    # training
    trainer = Trainer(energy_estimator, train_loader, eval_loader, params)
    trainer.train()
