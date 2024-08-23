#!/bin/sh

#SBATCH --partition=intelsr_short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source ~/.bashrc
conda deactivate
conda activate my_research_env

python ~/sp/main_roy.py