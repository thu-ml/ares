#!/bin/bash

source ~/begin04.sh
source activate pytorch04

python tune.py --data_path ~/cifar_data --dataset ${DATASET} --corruption_prob ${CPROB} --num_classes_pretrained_net 1000
