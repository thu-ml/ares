<div align="center">

# ARES 2.0: Robust Training for Image Classification

</div>

## Abstract

This repository contains the code for adversarial training on classification models, which is derived from [A Comprehensive Study on Robustness of Image Classification Models: Benchmarking and Rethinking](https://arxiv.org/abs/2302.14301), a Python library for adversarial machine learning research focusing on benchmarking adversarial robustness on image classification correctly and comprehensively. The project incorporates the widely adopted [timm](https://github.com/huggingface/pytorch-image-models) as its default classification library.

## Major features

- **Integration with [timm](https://github.com/huggingface/pytorch-image-models)**
  
  - Leverage various classification models from timm for adversarial training to achieve robustness across diverse model architectures.
- **State-of-the-art Models Available**
  
  - Some of the SOTA models are available from the model zoo, which are trained with the corresponding settings.
- **Multiple Augmentations**
  
  - Multiple augmentations are supported, including [Mixup](https://arxiv.org/abs/1710.09412), Label Smoothing, EMA and so on.
- **Distributed training and testing**
  
  - Pytorch distributed data-parallel training and testing are supported for faster training and testing.

## Preparation

**Dataset**

- We train our models with ImageNet dataset. Please download [ImageNet](https://www.image-net.org/) dataset first. The directories to the training and evaluation dataset should be assigned to `train_dir` and `eval_dir` in the `train_configs` files.

**Classification Model**

- Train classification models using [timm](https://github.com/huggingface/pytorch-image-models) or from your own model class.

## Getting Started

- We provide a command line interface to run adversarial training. For example, you can train a robust model of ResNet50 with the corresponding configuration:
  
  ```bash
  python -m torch.distributed.launch --nproc_per_node=<num-of-gpus-to-use> adversarial_training.py --configs=./train_configs/resnet50.yaml
  ```
- For distributed training and testing, you can also refer to the [run_train.sh](run_train.sh) for details.

## Results

- **Evaluation of some classification models**
  
  Attack settings: adversarial attack using PGD and autoattack with eps=4/255 under L $\infty$ norm.
  
  Dataset settings: randomly sampling 1000 data from ImageNet validation set.
  
  Model settings: adversarially trained on ImageNet training set.
  
  |    Model Name     |                                                                        Clean Accuracy                                                                         |                                                                                                           FGSM                                                                                                            |    PGD100     | AutoAttack | RobustBench |Checkpoints |
  |:---------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------:|:----:|:-------:|:-------:|
  | ResNet50 |67.0 | 44.5 | 38.7 | 34.1 | - | [Download](https://ml.cs.tsinghua.edu.cn/~xiaoyang/aresbench/ckpt-imagenet/ARES_ResNet50_AT.pth) |
  | ResNet101 | 71.0 | 51.3 | 46.5 | 42.2 | - | [Download](https://ml.cs.tsinghua.edu.cn/~xiaoyang/aresbench/ckpt-imagenet/ARES_ResNet101_AT.pth) |
  | ResNet152| 72.4 | 54.6 | 49.6 | 46.7 | - | [Download](https://ml.cs.tsinghua.edu.cn/~xiaoyang/aresbench/ckpt-imagenet/ARES_ResNet152_AT.pth) |
  | Wide-ResNet50 | 70.5 | 51.8 | 44.6 | 39.3 | - | [Download](https://ml.cs.tsinghua.edu.cn/~xiaoyang/aresbench/ckpt-imagenet/ARES_Wide_ResNet50_2_AT.pth) |
  | ConvNextS | 77.3 | 60.3 | 56.9 | 54.3 | - | [Download](https://ml.cs.tsinghua.edu.cn/~xiaoyang/aresbench/ckpt-imagenet/ARES_ConvNext_Small_AT.pth) |
  | ConvNextB| 77.2 | 62.2 | 59.0 | 56.8 | 55.82 | [Download](https://ml.cs.tsinghua.edu.cn/~xiaoyang/aresbench/ckpt-imagenet/ARES_ConvNext_Base_AT.pth) |
  | ConvNextL | 78.8 | 63.9 | 61.7 | 60.1 | 58.48 | [Download](https://ml.cs.tsinghua.edu.cn/~xiaoyang/aresbench/ckpt-imagenet/ARES_ConvNext_Large_AT.pth) |
  | ViTS| 70.7 | 51.3 | 47.5 | 43.7 | - | [Download](https://ml.cs.tsinghua.edu.cn/~xiaoyang/aresbench/ckpt-imagenet/ARES_ViT_small_patch16_224_AT.pth) |
  | ViTB | 74.7 | 55.9 | 52.2 | 49.7 | - | [Download](https://ml.cs.tsinghua.edu.cn/~xiaoyang/aresbench/ckpt-imagenet/ARES_ViT_base_patch16_224_AT.pth) |
  | SwinS | 76.6 | 61.5 | 58.4 | 55.6 | - | [Download](https://ml.cs.tsinghua.edu.cn/~xiaoyang/aresbench/ckpt-imagenet/ARES_Swin_small_patch4_window7_224_AT.pth) |
  | SwinB| 76.6 | 63.2 | 60.2 | 57.3 | 56.16 | [Download](https://ml.cs.tsinghua.edu.cn/~xiaoyang/aresbench/ckpt-imagenet/ARES_Swin_base_patch4_window7_224_AT.pth) |
  | SwinL | 79.7 | 65.9 | 63.9 | 62.3 | 59.56 | [Download](https://ml.cs.tsinghua.edu.cn/~xiaoyang/aresbench/ckpt-imagenet/ARES_Swin_large_patch4_window7_224_AT.pth) |


## Acknowledgement

Many thanks to these excellent open-source projects:

- [timm](https://github.com/huggingface/pytorch-image-models)
