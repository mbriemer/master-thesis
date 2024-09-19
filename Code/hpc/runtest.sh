#!/bin/bash

scp -r ./hpc/test_torch.py gpu:~/test/test_torch.py 
scp ./hpc/torch.sh gpu:~/test/torch.sh 
ssh gpu "sbatch ~/test/torch.sh" && ssh gpu