<div align="center">

# ARES 2.0: Robustness Evaluation for Image Classification

</div>

## Abstract
This repository contains the code to evaluate the adversarial robustness of classification models. 
We extend the original benchmark [Benchmarking Adversarial Robustness on Image Classification](http://openaccess.thecvf.com/content_CVPR_2020/papers/Dong_Benchmarking_Adversarial_Robustness_on_Image_Classification_CVPR_2020_paper.pdf) to 19 attacks and 65 models. 
The well-known [timm](https://github.com/huggingface/pytorch-image-models) project is used as the default classification library.

## Major features
- Built on PyTorch and Support of [timm](https://github.com/huggingface/pytorch-image-models)
  - Most classification models from timm can be used to conduct adversarial training. You can easily obtain robust models with different model architectures.

- Support [many attacks](https://github.com/Niculuse/ares2.0/tree/master/ares/attack) in various threat models.

- Provide ready-to-use pre-trained baseline models ([55 on ImageNet](https://github.com/Niculuse/ares2.0/blob/master/ares/model/imagenet_model_zoo.py) & [10 on CIFAR10](https://github.com/Niculuse/ares2.0/blob/master/ares/model/cifar_model_zoo.py)).

- Provide efficient & easy-to-use tools for evaluating classification models.

  
## Preparation
**Dataset**
- Support ImageNet and Cifar10 datasets for evaluation. For custom datasets, users should define their `torch.utils.data.Dataset` class and corresponding `transform`.

**Classification Model**
- Train classification models using [timm](https://github.com/huggingface/pytorch-image-models) or from your own model class.

## Getting Started
- Modify attack config files
  - We provide some common settings for all the adversarial attacks in a config file [attack_configs.py](https://github.com/Niculuse/ares2.0/blob/master/classification/attack_configs.py). Modify `attack_configs` dictionary according to your needs.
  - Define custom `torch.utils.data.Dataset` and `transform` and replace the original ones in [run_attack.py](https://github.com/Niculuse/ares2.0/tree/master/classification/run_attack.py) if a new dataset is evaluated.

- We provide a command line interface to run adversarial robustness evaluation. For example, you can evaluate an adversarially trained ResNet50 in our model zoo with PGD attack:
  ```bash
  python run_attack.py --gpu 0 --crop_pct 0.875 --input_size 224 --interpolation 'bilinear' --data_dir DATA_PATH --label_file LABEL_PATH --batchsize 20 --num_workers 16  --model_name 'resnet50_at' --attack_name 'pgd'
  ```

## Acknowledgement

Many thanks to these excellent open-source projects:

- [timm](https://github.com/huggingface/pytorch-image-models)
