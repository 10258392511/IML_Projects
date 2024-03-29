import argparse
import os
import numpy as np
import torch
import helpers.configs as configs
import helpers.pytorch_utils as ptu

from torch.utils.data import DataLoader
from helpers.utils import download_file, save_results
from helpers.models import FoodTaster, FoodDataset, predict
from tqdm import tqdm

model_params_url = "https://polybox.ethz.ch/index.php/s/WH5Vq4f8vV07iO2/download"
model_param_dir = "trained_params/"
model_param_filename = "food_taster.pt"

if __name__ == '__main__':
    """
    python ./main_pred.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=0)
    args = vars(parser.parse_args())
    # download model params
    model_param_path = os.path.join(model_param_dir, model_param_filename)
    if not os.path.isfile(model_param_path):
        download_file(model_params_url, model_param_dir, model_param_filename)
    else:
        print("Using saved parameters.")

    # test dataset & model
    all_params = {
        "batch_size": 256,
        "test_filename": "data/test_triplets.txt"
    }
    all_params.update(configs.configs_food_taster_param)
    all_params.update(args)

    test_dataset = FoodDataset(all_params["test_filename"], mode="test")
    test_loader = DataLoader(test_dataset, all_params["batch_size"], num_workers=all_params["num_workers"])

    for filename in os.listdir(model_param_dir):
        if filename.find(".pt") == -1:
            continue

        food_taster = FoodTaster(all_params).to(ptu.ptu_device)
        food_taster.load_state_dict(torch.load(os.path.join(model_param_dir, filename)))
        food_taster.eval()

        # predict
        with torch.no_grad():
            predictions = []
            pbar = tqdm(test_loader, total=len(test_loader))
            for i, (X1, X2, X3) in enumerate(pbar):
                # ### debug only ###
                # if i > 2:
                #     break
                # ###
                pred_batch = predict(food_taster, (X1, X2, X3))  # ndarray, (B,)
                predictions.append(pred_batch)

        predictions = np.concatenate(predictions, axis=-1)

        # save_results
        save_path = f"predictions/predictions_{filename.replace('.', '_')}.txt"
        save_results(predictions, save_path)
