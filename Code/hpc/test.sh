#!/bin/sh

#SBATCH --partition=intelsr_devel
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

source ~/.bashrc
conda deactivate
conda activate my_research_env

python ~/test/test.py