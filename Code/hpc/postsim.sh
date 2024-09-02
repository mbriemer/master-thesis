#!/bin/bash

DATE=$(date +%Y%m%d%H%M%S)

scp -r gpu:~/simres ../simres/simres_$DATE && ssh gpu 'rm ~/simres/*'

python ./sp/plot_npz.py $DATE