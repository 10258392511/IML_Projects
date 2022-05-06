import numpy as np
import os
import shutil

from helpers.utils import read_prediction, save_results, download_file, unzip_file


def read_from_directory_and_ensemble(dir_name):
    predictions = []
    for filename in os.listdir(dir_name):
        filename = os.path.join(dir_name, filename)
        if os.path.isfile(filename) and filename.find(".txt") >= 0:
            predictions.append(read_prediction(filename))

    assert len(predictions) % 2 == 1, "only support odd number of models"
    predictions = np.array(predictions)  # (N_pred, N_samples)
    ensemble = predictions.mean(axis=0) > 0.5

    return ensemble.astype(int)


def create_submission():
    temp_dir = "./submission_task_3"
    if os.path.isdir(temp_dir):
        user_in = input("Delete the old directory (Y/N): ")
        if user_in.lower() == "y":
            shutil.rmtree(temp_dir)

    os.mkdir(temp_dir)
    for filename in os.listdir("."):
        if filename.find(".py") >= 0 or filename.find(".md") >= 0:
            shutil.copy(filename, temp_dir)

    helpers_dir = os.path.join(temp_dir, "helpers")
    os.mkdir(helpers_dir)
    shutil.copytree("./helpers", helpers_dir, dirs_exist_ok=True)
    pycache_dir = os.path.join(helpers_dir, "__pycache__")
    if os.path.isdir(pycache_dir):
        shutil.rmtree(pycache_dir)
    shutil.make_archive("task3", "zip", temp_dir)


if __name__ == '__main__':
    """
    TODO
    Trained models to make ensemble are available at ...
    """
    url = "https://polybox.ethz.ch/index.php/s/m8yWi1I6HryPg0y/download"
    save_dir = "./predictions_downloaded"
    save_filename = "predictions.zip"
    save_path = os.path.join(save_dir, save_filename)
    download_file(url, save_dir, save_filename)
    unzip_file(save_path, ".")

    ensemble = read_from_directory_and_ensemble("./predictions")
    save_results(ensemble, "predictions.txt")

    create_submission()
