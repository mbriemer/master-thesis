#!/bin/sh

#SBATCH --partition=mlgpu_devel
#SBATCH --time=00:01:00
#SBATCH --gpus=1
#SBATCH --output="test/%j.out"

source ~/.bashrc
conda deactivate
conda activate mt
module load CUDA/12.4.0

python ~/test/test_torch.py