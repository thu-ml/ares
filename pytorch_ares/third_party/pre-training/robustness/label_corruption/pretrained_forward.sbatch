#!/bin/bash

source ~/begin04.sh
source activate pytorch04

echo "GF: ${GF}, CPROB: ${CPROB}, SEED: ${SEED}"
python train_forward.py ~/cifar_data ${DATASET} -s ./snapshots --log ./logs --gold_fraction ${GF} --corruption_prob ${CPROB} --corruption_type "unif" --seed ${SEED} --use_pretrained_model
