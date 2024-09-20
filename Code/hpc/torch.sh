#!/bin/sh

#SBATCH --partition=mlgpu_short
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --output="simres/%j.out"

source ~/.bashrc
conda deactivate
conda activate mt
module load CUDA/12.4.0

python $HOME/torch/main_roy.py