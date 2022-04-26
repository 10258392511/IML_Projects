import argparse
import os
import time
import helpers.configs as configs
import helpers.pytorch_utils as ptu

from helpers.utils import download_file, unzip_file, create_param_save_path, create_log_dir
from torch.utils.data import DataLoader
from helpers.models import FoodDataset, FoodTaster
from helpers.trainer import FoodTasterTrainer

dataset_url = "https://polybox.ethz.ch/index.php/s/39L5nDkzCNEhJ6J/download"


if __name__ == '__main__':
    """
    Run at root of the project (i.e. from this script's location)
    python ./main_train.py --batch_size 64 --epochs 10
    """
    # download dataset
    data_dir = "./data"
    zip_filename = "food.zip"
    zip_path = os.path.join(data_dir, zip_filename)
    if not os.path.isdir(data_dir):
        download_file(dataset_url, data_dir, zip_filename)
        unzip_file(zip_path, ".")
        os.remove(zip_path)
        print("Done retrieving dataset!")
    else:
        print("Dataset already downloaded.")

    # training
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    args = vars(parser.parse_args())

    time_stamp = f"{time.time()}".replace(".", "_")
    all_params = {
        "if_notebook": False,
        "train_filename": "data/train_triplets.txt",
        "test_filename": "data/test_triplets.txt"
    }
    for arg_dict in (args, configs.configs_trainer_param, configs.configs_food_taster_param):
        all_params.update(arg_dict)

    log_dir_args = {
        "batch_size": all_params["batch_size"],
        "epochs": all_params["epochs"],
        "lr": all_params["opt_args"]["args"]["lr"],
        "feature_dim": all_params["feature_dim"],
        "alpha": all_params["alpha"]
    }
    dir_name = create_log_dir(time_stamp, log_dir_args)
    all_params.update({
        "log_dir": f"run/food_taster/{dir_name}",
        "param_save_dir": f"params/food_taster/{dir_name}"
    })

    train_dataset = FoodDataset(all_params["train_filename"], mode="train")
    eval_dataset = FoodDataset(all_params["train_filename"], mode="val")
    test_dataset = FoodDataset(all_params["test_filename"], mode="test")

    train_loader = DataLoader(train_dataset, batch_size=all_params["batch_size"], num_workers=all_params["num_workers"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=all_params["batch_size"], num_workers=all_params["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=all_params["batch_size"], num_workers=all_params["num_workers"])

    food_taster = FoodTaster(all_params).to(ptu.ptu_device)
    trainer = FoodTasterTrainer(food_taster, train_loader, eval_loader, all_params)
    trainer.train()
