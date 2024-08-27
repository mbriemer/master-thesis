#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

scp -r ./sp/ gpu:~
scp ./hpc/sp.sh gpu:~/sp/sp.sh
ssh gpu "mkdir -p ~/simres_${TIMESTAMP}"
ssh gpu "cd ~/simres_${TIMESTAMP}"
ssh gpu "sbatch ../sp/sp.sh"