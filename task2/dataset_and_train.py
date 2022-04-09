import argparse
import time
import os
import torch
import helpers.configs as configs

from torch.utils.data import DataLoader, random_split
from helpers.preprocess import Task2Dataset
from helpers.models import SequenceModel
from helpers.trainers import SeqModelTrainer

if __name__ == '__main__':
    """
    python dataset_and_train.py --log_dir ./run --model_save_dir ./params
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--test_split_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cls_weight", type=float, default=20.)
    parser.add_argument("--last_cls_weight", type=float, default=15.)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--model_save_dir", required=True)
    parser.add_argument("--if_notebook", action="store_true")
    args = parser.parse_args()
    params = vars(args)
    params.update(configs.conf_auc_params)
    params.update(configs.conf_seq_model_params)
    params.update(configs.conf_seq_model_opt_params)

    timestamp = f"{time.time()}".replace(".", "_")
    params["log_dir"] = os.path.join(params["log_dir"], timestamp)
    params["model_save_dir"] = os.path.join(params["model_save_dir"], timestamp)

    train_filename = "data/train.pkl"
    test_filename = "data/test.pkl"
    train_dataset = Task2Dataset(train_filename)
    test_dataset = Task2Dataset(test_filename)
    num_val_samples = int(len(train_dataset) * params["test_split_ratio"])
    train_dataset, val_dataset = random_split(train_dataset, [len(train_dataset) - num_val_samples, num_val_samples],
                                              generator=torch.Generator().manual_seed(params["seed"]))
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"])

    seq_model = SequenceModel(configs.conf_seq_model_params).to(configs.conf_device)
    trainer = SeqModelTrainer(seq_model, train_loader, val_loader, params)
    trainer.train()
