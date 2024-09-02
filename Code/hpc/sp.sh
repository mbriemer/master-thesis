#!/bin/sh

#SBATCH --partition=intelsr_medium
#SBATCH --time=7:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output="simres/job_%j.out"

source ~/.bashrc
conda deactivate
conda activate my_research_env

python $HOME/sp/main_roy.py