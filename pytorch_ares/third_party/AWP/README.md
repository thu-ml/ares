# Adversarial Weight Perturbation Helps Robust Generalization

Code for NeurIPS 2020 "[Adversarial Weight Perturbation Helps Robust Generalization](https://arxiv.org/pdf/2004.05884.pdf)" by [Dongxian Wu](https://scholar.google.com/citations?user=ZQzqQqwAAAAJ&hl=en&oi=ao), [Shu-Tao Xia](https://scholar.google.com/citations?user=koAXTXgAAAAJ&hl=en&oi=ao), and [Yisen Wang](https://sites.google.com/site/csyisenwang/).

## News

10/13/2020 - Our code and paper are released.

## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.7.3
- torch = 1.2.0
- torchvision = 0.4.0

## What is in this repository

Codes for our AWP-based adversarial training (AT-AWP) are in `at-awp`, and those for AWP-based TRADES (TRADES-AWP) are in `./trades-awp`:
- In `./at-awp`, the codes for CIFAR-10, CIFAR-100, and SVHN are in `train_cifar10.py`, `train_cifar100.py`, `train_svhn.py` respectively.
- In `./trades-awp`, the codes for CIFAR-10 and CIFAR-100 are in `train_trades_cifar.py`.

The checkpoints can be found in [Google Drive](https://drive.google.com/drive/folders/1K1hvOZ4qTWYil3hv32IDoyr_xGjf4ZN-?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1ZtY3RweP10m_ev0XF5zB6A)(pw: 8tsv).

## How to use it

For AT-AWP with a PreAct ResNet-18 on CIFAR-10 under L_inf threat model (8/255), run codes as follows, 
```
python train_cifar10.py --data-dir DATASET_DIR
```
where `$DATASET_DIR` is the path to the dataset. 

For TRADES-AWP with a WRN-34-10 on CIFAR10 under L_inf threat model (8/255), run codes as follows,
```
python train_trades_cifar.py --data CIFAR10 --data-path DATASET_DIR
```
## The Leaderboard Under Auto Attack

To verify the effectiveness of AWP further, we evaluate the robustness under a stronger attack, auto-attack [3]. Here we only list Top 10 results on the leadboard (up to 10/13/2020) and our results. Compared with the leadboard results, AWP can boost the robustness of the AT and its variants (TRADES[2], MART[4], Pre-training[5], RST[6], etc.), ranking 1st on both with and without data. Even some AWP-based methods without additional data can surpass the results under additional data. 

More results can be found in [`./auto-attacks`](https://github.com/csdongxian/AWP/tree/main/auto_attacks)

|#    |method / paper           |model     |architecture |clean         |report. |AA  |
|:---:|---|:---:|:---:|---:|---:|---:|
|**-**| **RST-AWP (ours)**‡| [*downloads*](https://drive.google.com/file/d/1sSjh4i2imdoprw_JcPj2cZzrJm0RIRI6/view?usp=sharing)| **WRN-28-10**| **88.25**| - | **60.04**|
|**1**| [(Wu et al., 2020)](https://arxiv.org/abs/2010.01279)‡| *available*| WRN-34-15| 85.60| 59.78| 59.78|
|**2**| [(Carmon et al., 2019)](https://arxiv.org/abs/1905.13736) **RST**‡| *available*| WRN-28-10| 89.69| 62.5| 59.53|
|**-**| **Pre-training-AWP (ours)**‡| [*downloads*](https://drive.google.com/file/d/1xwisiNlxqoODnkJ2pP4g8wHD3tBgk7AM/view?usp=sharing)| **WRN-28-10**| **88.33**| - | **57.39**|
|**3**| [(Sehwag et al., 2020)](https://github.com/fra31/auto-attack/issues/7)‡| *available*| WRN-28-10| 88.98| -| 57.14|
|**4**| [(Wang et al., 2020)](https://openreview.net/forum?id=rklOg6EFwS)‡| *available*| WRN-28-10| 87.50| 65.04| 56.29|
|**-**| **TRADES-AWP (ours)**| [*downloads*](https://drive.google.com/file/d/1hlVTLZkveYGWpE9-46Wp5NVZt1slz-1T/view?usp=sharing)| **WRN-34-10**| **85.36**| - | **56.17**|
|**5**| [(Alayrac et al., 2019)](https://arxiv.org/abs/1905.13725)‡| *available*| WRN-106-8| 86.46| 56.30| 56.03|
|**6**| [(Hendrycks et al., 2019)](https://arxiv.org/abs/1901.09960) **Pre-training**‡| *available*| WRN-28-10| 87.11| 57.4| 54.92|
|**-**| **MART-AWP (ours)**| [*downloads*](https://drive.google.com/file/d/1RwHjupK2dshNHm_4fK3h1-Ys0RckhXvH/view?usp=sharing)| **WRN-34-10**| **84.43**| - | **54.23**|
|**-**| **AT-AWP (ours)**| [*downloads*](https://drive.google.com/file/d/1iNfy-yTUEPuSK2uHO5tiFdEBehmQWbbN/view?usp=sharing)| **WRN-34-10**| **85.36**| - | **53.97**|
|**7**| [(Pang et al., 2020b)](https://arxiv.org/abs/2002.08619)| *available*| WRN-34-20| 85.14| -| 53.74|
|**8**| [(Zhang et al., 2020b)](https://arxiv.org/abs/2002.11242)| *available*| WRN-34-10| 84.52| 54.36| 53.51|
|**9**| [(Rice et al., 2020)](https://arxiv.org/abs/2002.11569) **AT**| *available*| WRN-34-20| 85.34| 58| 53.42|
|**10**| [(Huang et al., 2020)](https://arxiv.org/abs/2002.10319)\*| *available*| WRN-34-10| 83.48| 58.03| 53.34|

## Citing this work
```
@inproceedings{wu2020adversarial,
    title={Adversarial Weight Perturbation Helps Robust Generalization},
    author={Dongxian Wu and Shu-Tao Xia and Yisen Wang},
    booktitle={NeurIPS},
    year={2020}
}
```


## Reference Code
[1] AT: https://github.com/locuslab/robust_overfitting

[2] TRADES: https://github.com/yaodongyu/TRADES/

[3] AutoAttack: https://github.com/fra31/auto-attack

[4] MART: https://github.com/YisenWang/MART

[5] Pre-training: https://github.com/hendrycks/pre-training

[6] RST: https://github.com/yaircarmon/semisup-adv
