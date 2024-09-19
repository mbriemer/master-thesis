#!/bin/sh

#SBATCH --partition=intelsr_short
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output="simres/%j.out"

source ~/.bashrc
conda deactivate
conda activate my_research_env

python $HOME/sp/diagonal_loss_plots.py