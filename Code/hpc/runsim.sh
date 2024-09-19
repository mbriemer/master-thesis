#!/bin/bash

scp -r ./torch/ gpu:~
scp ./hpc/torch.sh gpu:~/torch/torch.sh
ssh gpu "sbatch ~/torch/torch.sh" && ssh gpu