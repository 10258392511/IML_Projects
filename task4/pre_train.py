import argparse
import helpers.configs as configs

from datetime import datetime
from torch.utils.data import DataLoader
from helpers.dataset import MoleculeDataset
from helpers.modules import EnergyEstimator
from helpers.trainer import Trainer
from helpers.utils import create_log_dir


if __name__ == '__main__':
    """
    python pre_train.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    params = vars(parser.parse_args())

    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    log_keys = ["batch_size", "epochs"]
    log_params = {key: params[key] for key in log_keys}
    log_dir = create_log_dir(time_stamp, log_params)
    params.update({
        "log_dir": f"run/pre_train/{log_dir}",
        "param_save_dir": f"params/pre_train/{log_dir}",
        "if_notebook": False
    })
    params.update(configs.config_train_params)

    # dataset
    pre_train_ds = MoleculeDataset("pre-train", seed=params["seed"], vocab_path="info/vocab.pkl")
    pre_eval_ds = MoleculeDataset("pre-eval", seed=params["seed"], vocab_path="info/vocab.pkl")
    pre_train_loader = DataLoader(pre_train_ds, batch_size=params["batch_size"], shuffle=True)
    pre_eval_loader = DataLoader(pre_eval_ds, batch_size=params["batch_size"])

    # model
    energy_estimator_params = {
        "seq_encoder_params": configs.config_seq_encoder_params,
        "mlp_params": configs.config_mlp_params,
        "fusion_params": configs.config_fusion_params
    }

    energy_estimator = EnergyEstimator(energy_estimator_params).to(configs.config_device)

    # training
    pre_trainer = Trainer(energy_estimator, pre_train_loader, pre_eval_loader, params)
    pre_trainer.train()
