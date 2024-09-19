#!/bin/sh

#SBATCH --partition=mlgpu_short
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --output="simres/%j.out"

source ~/.bashrc
conda deactivate
conda activate mt

python $HOME/torch/main_roy.py