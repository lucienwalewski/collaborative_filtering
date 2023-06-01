#!/bin/bash

#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:8G
#SBATCH --time=4:00:00

python3 mlp/train.py "$@"
 