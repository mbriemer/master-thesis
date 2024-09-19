#!/bin/bash

scp -r ./sp/ gpu:~
scp ./hpc/sp.sh gpu:~/sp/sp.sh
ssh gpu "sbatch ~/sp/sp.sh" && ssh gpu