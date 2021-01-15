#!/bin/bash
echo -e "\033[33mRunning 'python3 ../example/cifar10/pgd_at.py'\033[0m"
python3 ../example/cifar10/pgd_at.py
echo -e "\033[33mRunning 'python3 ../example/cifar10/wideresnet_trades.py'\033[0m"
python3 ../example/cifar10/wideresnet_trades.py
echo -e "\033[33mRunning 'python3 ../example/cifar10/feature_scatter.py'\033[0m"
python3 ../example/cifar10/feature_scatter.py
echo -e "\033[33mRunning 'python3 ../example/cifar10/robust_overfitting.py'\033[0m"
python3 ../example/cifar10/robust_overfitting.py
echo -e "\033[33mRunning 'python3 ../example/cifar10/rst.py'\033[0m"
python3 ../example/cifar10/rst.py
echo -e "\033[33mRunning 'python3 ../example/cifar10/fast_at.py'\033[0m"
python3 ../example/cifar10/fast_at.py
echo -e "\033[33mRunning 'python3 ../example/cifar10/at_he.py'\033[0m"
python3 ../example/cifar10/at_he.py
echo -e "\033[33mRunning 'python3 ../example/cifar10/pre_training.py'\033[0m"
python3 ../example/cifar10/pre_training.py
echo -e "\033[33mRunning 'python3 ../example/cifar10/mmc.py'\033[0m"
python3 ../example/cifar10/mmc.py
echo -e "\033[33mRunning 'python3 ../example/cifar10/awp.py'\033[0m"
python3 ../example/cifar10/awp.py
echo -e "\033[33mRunning 'python3 ../example/cifar10/hydra.py'\033[0m"
python3 ../example/cifar10/hydra.py
echo -e "\033[33mRunning 'python3 ../example/cifar10/label_smoothing.py'\033[0m"
python3 ../example/cifar10/label_smoothing.py
echo -e "\033[33mRunning 'python3 ../example/cifar10/fast_at.py'\033[0m"
python3 ../example/cifar10/fast_at.py
echo -e "\033[33mRunning 'python3 ../example/cifar10/free_at.py'\033[0m"
python3 ../example/cifar10/free_at.py
echo -e "\033[33mRunning 'python3 ../example/imagenet/fast_at.py'\033[0m"
python3 ../example/imagenet/fast_at.py
echo -e "\033[33mRunning 'python3 ../example/imagenet/free_at.py'\033[0m"
python3 ../example/imagenet/free_at.py
echo -e "\033[33mRunning 'python3 -m ares.dataset.cifar10'\033[0m"
python3 -m ares.dataset.cifar10
echo -e "\033[33mRunning 'python3 -m ares.dataset.imagenet'\033[0m"
python3 -m ares.dataset.imagenet
