import argparse
import os
import time
import helpers.configs as configs
import helpers.pytorch_utils as ptu

from helpers.utils import download_file, unzip_file, create_param_save_path
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
    args = vars(parser.parse_args())

    time_stamp = f"{time.time()}".replace(".", "_")
    all_params = {
        "log_dir": f"run/food_taster/{time_stamp}",
        "param_save_dir": f"params/food_taster/{time_stamp}",
        "if_notebook": False,
        "train_filename": "data/train_triplets.txt",
        "test_filename": "data/test_triplets.txt"
    }
    for arg_dict in (args, configs.configs_trainer_param, configs.configs_food_taster_param):
        all_params.update(arg_dict)

    train_dataset = FoodDataset(all_params["train_filename"], mode="train")
    eval_dataset = FoodDataset(all_params["train_filename"], mode="val")
    test_dataset = FoodDataset(all_params["test_filename"], mode="test")

    train_loader = DataLoader(train_dataset, batch_size=all_params["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=all_params["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=all_params["batch_size"])

    food_taster = FoodTaster(all_params).to(ptu.ptu_device)
    trainer = FoodTasterTrainer(food_taster, train_loader, eval_loader, all_params)
    trainer.train()
