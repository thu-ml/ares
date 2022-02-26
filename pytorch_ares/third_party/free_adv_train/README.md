# Free Adversarial Training 
This repository belongs to the [Free Adversarial Training](https://arxiv.org/abs/1904.12843 "Free Adversarial Training") paper.
The implementation is inspired by [CIFAR10 Adversarial Example Challenge](https://github.com/MadryLab/cifar10_challenge "Madry's CIFAR10 Challenge") so to them we give the credit.
This repo is for the CIFAR-10 and CIFAR-100 datasets and is in Tensorflow. Our Free-m models can acheive comparable performance with conventional PGD adversarial training at a fraction of the time. 

**__News!__**: We have released our [ImageNet implementation of Free adversarial training in Pytorch](https://github.com/mahyarnajibi/FreeAdversarialTraining) !


###### CIFAR-10 WRN 32-10 (L-inf epsilon=8):

| Model | Natural | PGD-100 | CW-100 | 10 restart PGD-20 | train-time (min) |
| --- | --- | --- | --- | --- | --- |
| Natrual | 95.01 | 0.00 | 0.00| 0.00 | 780 |
| Free-2 | 91.45 | 33.20 | 34.57 | 33.41 | 816 |
| Free-4 | 87.83 | 40.35 | 41.96 | 40.73 | 800 |
| **Free-8** | **85.96** | **46.19** | **46.60** | **46.33** | **785** |
| Free-10 |83.94 | 45.79 | 45.86 | 45.94 | 785 |
|Madry 7-PGD (public model) | 87.25 | 45.29 | 46.52 | 45.53 | 5418 |

###### CIFAR-100 WRN 32-10 (L-inf epsilon=8):
| Model | Natural | PGD-20 | PGD-100  | train-time (min) |
| --- | --- | --- | --- | --- |
| Natrual | 78.84 | 0.00 | 0.00 | 811 |
| Free-2 | 69.20 | 15.37 | 14.86 | 816 |
| Free-4 | 65.28 | 20.64 | 20.15 | 767 |
| **Free-8** | **62.13** | **25.88** | **25.58** | **780** |
| Free-10 | 59.27 | 25.15 | 24.88 | 776 |
| Madry 2-PGD trained | 67.94 | 17.08 | 16.50 | 2053 |
| Madry 7-PGD trained | 59.87 | 22.76 | 22.52 | 5157 |


## Demo
To train a new robust model for free! run the following command specifying the replay parameter `m`:

```bash
python free_train.py -m 8
```

To evaluate a robust model using PGD-20 with 2 random restarts run:

```bash
python multi_restart_pgd_attack.py --model_dir $MODEL_DIR --num_restarts 2
```
Note that if you have trained a CIFAR-100 model, even for evaluation, you should pass the dataset argument. For example:
```bash
python multi_restart_pgd_attack.py --model_dir $MODEL_DIR_TO_CIFAR100 --num_restarts 2 -d cifar100
```

## Requirements 
To install all the requirements plus tensorflow for multi-gpus run: (Inspired By [Illarion ikhlestov](https://github.com/ikhlestov/vision_networks "Densenet Implementation") ) 

```bash
pip install -r requirements/gpu.txt
```

Alternatively, to install the requirements plus tensorflow for cpu run: 
```bash
pip install -r requirements/cpu.txt
```

To prepare the data, please see [Datasets section](https://github.com/ashafahi/free_adv_train/tree/master/datasets "Dataset readme").

If you find the paper or the code useful for your study, please consider citing the free training paper:
```bash
@article{shafahi2019adversarial,
  title={Adversarial Training for Free!},
  author={Shafahi, Ali and Najibi, Mahyar and Ghiasi, Amin and Xu, Zheng and Dickerson, John and Studer, Christoph and Davis, Larry S and Taylor, Gavin and Goldstein, Tom},
  journal={arXiv preprint arXiv:1904.12843},
  year={2019}
}
```
