import subprocess
import argparse

# The generated bash script should run in submission/


def make_bash_script(hyper_param_dict: dict):
    bash_script = f"""#!/bin/bash
#SBATCH --output=logs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
eval "$(conda shell.bash hook)"
conda activate deep_learning
# cd ..

python ./main_train.py --batch_size {hyper_param_dict["batch_size"]} --epochs {hyper_param_dict["epochs"]} --num_workers 8"""

    return bash_script


def create_filename(hyper_param_dict: dict):
    filename = ""
    for key, val in hyper_param_dict.items():
        filename += f"{key}_{val}_"

    filename = filename[:-1].replace(".", "_") + ".sh"

    return filename


if __name__ == '__main__':
    """
    python ./generate_bash_main_train.py --set_num 1
    """
    hyper_params = dict()
    # set 1
    hyper_params[1] = [{"batch_size": 128, "epochs": 10}]

    parser = argparse.ArgumentParser()
    parser.add_argument("--set_num", type=int, choices=hyper_params.keys(), required=True)

    args = parser.parse_args()

    hyper_params_list = hyper_params[args.set_num]
    for hyper_param_dict_iter in hyper_params_list:
        filename = create_filename(hyper_param_dict_iter)
        bash_script = make_bash_script(hyper_param_dict_iter)
        subprocess.run(f"echo '{bash_script}' > {filename}", shell=True)
        # print(f"{filename}")
        # print(f"{bash_script}")
        # print("-" * 50)
