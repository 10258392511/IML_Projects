#!/bin/bash
#SBATCH --output=logs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
eval "$(conda shell.bash hook)"
conda activate deep_learning
# cd ..

python ./main_pred.py --num_workers 8
