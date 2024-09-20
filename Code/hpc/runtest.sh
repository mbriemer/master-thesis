#!/bin/bash

scp -r ./hpc/test_torch.py gpu:~/test/test_torch.py 
scp ./hpc/test.sh gpu:~/test/test.sh 
ssh gpu "sbatch ~/test/test.sh" && ssh gpu