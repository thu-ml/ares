#!/bin/bash

cd "$(dirname "$0")"

python3 -m realsafe.benchmark.distortion_cli --method fgsm --dataset cifar10 --offset 0 --count 1000 --output fgsm.npy ../example/cifar10/resnet56.py --distortion 0.1 --goal ut --distance-metric l_inf --batch-size 100 --logger
python3 -m realsafe.benchmark.distortion_cli --method bim --dataset cifar10 --offset 0 --count 1000 --output bim.npy ../example/cifar10/resnet56.py --distortion 0.1 --goal ut --distance-metric l_inf --batch-size 100 --iteration 10 --logger
python3 -m realsafe.benchmark.distortion_cli --method pgd --dataset cifar10 --offset 0 --count 1000 --output pgd.npy ../example/cifar10/resnet56.py --distortion 0.1 --goal ut --distance-metric l_inf --batch-size 100 --iteration 10 --rand-init-magnitude 0.05 --logger
python3 -m realsafe.benchmark.distortion_cli --method mim --dataset cifar10 --offset 0 --count 1000 --output mim.npy ../example/cifar10/resnet56.py --distortion 0.1 --goal ut --distance-metric l_inf --batch-size 100 --iteration 10 --decay-factor 1.0 --logger
python3 -m realsafe.benchmark.distortion_cli --method cw --dataset cifar10 --offset 0 --count 1000 --output cw.npy ../example/cifar10/resnet56.py --goal ut --distance-metric l_2 --batch-size 100 --iteration 20 --cs 1.0 --learning-rate 0.001 --logger
python3 -m realsafe.benchmark.distortion_cli --method deepfool --dataset cifar10 --offset 0 --count 1000 --output deepfool.npy ../example/cifar10/resnet56.py --goal ut --distance-metric l_2 --batch-size 100 --iteration 50 --overshot 0.2 --logger
python3 -m realsafe.benchmark.distortion_cli --method nes --dataset cifar10 --offset 0 --count 1 --output nes.npy ../example/cifar10/resnet56.py --goal ut --distance-metric l_inf --batch-size 100 --samples-per-draw 100 --max-queries 20000 --distortion 0.1 --nes-lr-factor 0.15 --nes-min-lr-factor 0.015 --sigma 0.001 --lr-tuning --plateau-length 20 --logger
python3 -m realsafe.benchmark.distortion_cli --method spsa --dataset cifar10 --offset 0 --count 1 --output spsa.npy ../example/cifar10/resnet56.py --goal ut --distance-metric l_inf --batch-size 100 --samples-per-draw 100 --max-queries 20000 --distortion 0.1 --spsa-lr-factor 0.3 --sigma 0.001 --logger
python3 -m realsafe.benchmark.distortion_cli --method nattack --dataset cifar10 --offset 0 --count 1 --output nattack.npy ../example/cifar10/resnet56.py --goal ut --distance-metric l_inf --batch-size 100 --samples-per-draw 100 --max-queries 20000 --distortion 0.1 --sigma 0.1 --lr 0.02 --logger
