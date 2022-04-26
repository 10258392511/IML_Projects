#!/bin/bash
#SBATCH --output=logs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=5
eval "$(conda shell.bash hook)"
conda activate deep_learning
# cd ..

python ./main.py --num_workers 16
