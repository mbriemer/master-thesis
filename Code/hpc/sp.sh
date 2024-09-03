#!/bin/sh

#SBATCH --partition=intelsr_medium
#SBATCH --time=05:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output="simres/%j.out"

source ~/.bashrc
conda deactivate
conda activate my_research_env

python $HOME/sp/main_roy.py