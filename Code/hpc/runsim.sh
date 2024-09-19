#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <torch or sp>"
    exit 1
fi

scp -r ./$1/ gpu:~
scp ./hpc/$1.sh gpu:~/$1/$1.sh
ssh gpu "sbatch ~/$1/$1.sh" && ssh gpu