# ImageNet Robust Models

This repository contains the robust models trained on ImageNet, and the scripts for robustness evaluation.

The benchmarked results have been contained in ARES-Bench. 

# Usage

First, clone the repository locally:
```
git clone https://github.com/alibaba/easyrobust.git
cd easyrobust
pip install -r requirements.txt
```
Then test runing on ImageNet Validation set:
```
python robustness_validation.py --model=resnet50 --interpolation=3 --imagenet_val_path=/path/to/ILSVRC/Data/CLS-LOC/val
```
The trained models will be downloaded automaticly. If you want to download the checkpoints manually, check the urls in [utils.py](https://github.com/alibaba/easyrobust/blob/main/utils.py).

The code supports [Stylized-ImageNet](https://github.com/rgeirhos/Stylized-ImageNet), [ImageNet-V2](https://github.com/modestyachts/ImageNetV2), [ImageNet-R](https://github.com/hendrycks/imagenet-r), [ImageNet-A](https://github.com/hendrycks/natural-adv-examples), [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch), [ObjectNet](https://objectnet.dev/), [ImageNet-C](https://github.com/hendrycks/robustness), [AutoAttack](https://github.com/fra31/auto-attack) evaluation. See [test_example.sh](https://github.com/alibaba/easyrobust/blob/main/test_example.sh) for details. 

## Adversarially robust models
18 Adversarially trained models are opened in `utils.py`. 

## Non-Adversarially robust models

We collect some non-adversarially robust models based on resnet50. To test these models, replace the [this line](https://github.com/alibaba/easyrobust/blob/db87c8f26a2b722ba5af1de4e6b9aebba76de6de/utils.py#L5) with following urls:

| Method   |  Architecture  | weights |
|:-------:|:--------:|:--------:|
| `SIN` |  resnet50 | http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/SIN.pth |
| `ANT` |  resnet50 | http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/ANT3x3_Model.pth |
| `Augmix` |  resnet50 | http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/augmix.pth |
| `DeepAugment` |  resnet50 | http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/deepaugment.pth |
| `DebiasedCNN` |  resnet50 | http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/imagenet_pretrained_models/clean_models/res50-debiased.pth |
