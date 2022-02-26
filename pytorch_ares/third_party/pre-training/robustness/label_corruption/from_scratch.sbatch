#!/bin/bash

source ~/begin04.sh
source activate pytorch04

python train.py --data_path ~/cifar_data --dataset ${DATASET} --corruption_prob ${CPROB}
