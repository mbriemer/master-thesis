#!/bin/sh

#SBATCH --partition=intelsr_devel
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda deactivate
conda activate my_research_env

python test.py