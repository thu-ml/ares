# Evaluation under AutoAttack

To verify the effectiveness of AWP further, we evaluate the robustness under a stronger attack, [auto-attack](https://github.com/fra31/auto-attack). 
The results up to 2020/10/13 can be seen below.

## CIFAR-10 - Linf
The robust accuracy is evaluated at `eps = 8/255`.\
**Note**: ‡ indicates models which exploit additional data for training (e.g. unlabeled data, pre-training).

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
|**11**| [(Zhang et al., 2019b)](https://arxiv.org/abs/1901.08573) **TRADES**\*| *available*| WRN-34-10| 84.92| 56.43| 53.08|
|**12**| [(Qin et al., 2019)](https://arxiv.org/abs/1907.02610v2)| *available*| WRN-40-8| 86.28| 52.81| 52.84|
|**13**| [(Chen et al., 2020a)](https://arxiv.org/abs/2003.12862)| *available*| RN-50 (x3)| 86.04| 54.64| 51.56|
|**14**| [(Chen et al., 2020b)](https://github.com/fra31/auto-attack/issues/26)| *available*| WRN-34-10| 85.32| 51.13| 51.12|
|**15**| [(Sitawarin et al., 2020)](https://github.com/fra31/auto-attack/issues/23)| *available*| WRN-34-10| 86.84| 50.72| 50.72|

## CIFAR-100 - Linf
The robust accuracy is computed at `eps = 8/255` in the Linf-norm.\
**Note**: ‡ indicates models which exploit additional data for training (e.g. unlabeled data, pre-training).

|#    |method / paper  |model     |architecture |clean         |report. |AA  |
|:---:|---|:---:|:---:|---:|---:|---:|
|**-**| **AT-AWP (ours)**| [*downloads*](https://drive.google.com/file/d/1aUQ3Udbn-zfQENwHRe8JsmkxjIcq0zVU/view?usp=sharing)| **WRN-34-10**| **60.38**| - | **28.86**|
|**-**| **TRADES-AWP (ours)**| [*downloads*](https://drive.google.com/file/d/1D-QCH-0ShtFo0s6gke5y6Ix7_x6k4Bys/view?usp=sharing)| **WRN-34-10**| **60.17**| - | **28.80**|
|**1**| [(Hendrycks et al., 2019)](https://arxiv.org/abs/1901.09960)‡| *available*| WRN-28-10| 59.23| 33.5| 28.42|
|**2**| [(Chen et al., 2020b)](https://github.com/fra31/auto-attack/issues/26)| *available*| WRN-34-10| 62.15| -| 26.94|
|**-**| **AT-AWP (ours)**| [*downloads*](https://drive.google.com/file/d/1IlrhlQyvNmlgnkGILTqt1yo8kHuooI8B/view?usp=sharing)| **RN-18**| **53.81**| **30.71**| **25.34**|
|**3**| [(Sitawarin et al., 2020)](https://github.com/fra31/auto-attack/issues/22)| *available*| WRN-34-10| 62.82| 24.57| 24.57|
|**4**| [(Rice et al., 2020)](https://arxiv.org/abs/2002.11569) **AT**| *available*| RN-18| 53.83| 28.1| 18.95|


## CIFAR-10 - L2
The robust accuracy is computed at `eps = 0.5` in the L2-norm.\
**Note**: ‡ indicates models which exploit additional data for training (e.g. unlabeled data, pre-training).

|#    |method / paper  |model     |architecture |clean         |report. |AA  |
|:---:|---|:---:|:---:|---:|---:|---:|
|**-**| **TRADES-AWP (ours)**| [*downloads*](https://drive.google.com/file/d/1D-QCH-0ShtFo0s6gke5y6Ix7_x6k4Bys/view?usp=sharing) | **WRN-34-10**| **88.51**| - | **73.66**|
|**1**| [(Augustin et al., 2020)](https://arxiv.org/abs/2003.09461)‡| *authors*| RN-50| 91.08| 73.27| 72.91|
|**-**| **AT-AWP (ours)**| [*downloads*](https://drive.google.com/file/d/1aUQ3Udbn-zfQENwHRe8JsmkxjIcq0zVU/view?usp=sharing) | **WRN-34-10**| **92.58**| - | **72.87**|
|**-**| **AT-AWP (ours)**| [*downloads*](https://drive.google.com/file/d/1iNfy-yTUEPuSK2uHO5tiFdEBehmQWbbN/view?usp=sharing) | **RN-18**| **90.11**| - | **70.31**|
|**2**| [(Engstrom et al., 2019)](https://github.com/MadryLab/robustness)| *available*| RN-50| 90.83| 70.11| 69.24|
|**3**| [(Rice et al., 2020)](https://arxiv.org/abs/2002.11569) **AT**| *available*| RN-18| 88.67| 71.6| 67.68|
|**4**| [(Rony et al., 2019)](https://arxiv.org/abs/1811.09600)| *available*| WRN-28-10| 89.05| 67.6| 66.44|
|**5**| [(Ding et al., 2020)](https://openreview.net/forum?id=HkeryxBtPB)| *available*| WRN-28-4| 88.02| 66.18| 66.09|

# How to evaluate under auto-attacks

### Installation

```
pip install git+https://github.com/fra31/auto-attack
```

### Run the evaluation
Since the source codes of AT and TRADES apply different preprocessing method, we should select one in evaluation, i.e., `--preprocess 'meanstd'` for AT, `--preprocess '01'` for TRADES. `$CKPT_DIR` is the path to the checkpoint. 
Thus, evaluate the robustness of models trained using AT-AWP as follows,
- For the SOTA result ([checkpoint](https://drive.google.com/file/d/1sSjh4i2imdoprw_JcPj2cZzrJm0RIRI6/view?usp=sharing)) with additional data on CIFAR10 under L_inf (8/255), run
```
python eval.py --arch WideResNet28 --checkpoint CKPT_DIR --data CIFAR10 --preprocess '01'
```
- For the SOTA result ([checkpoint](https://drive.google.com/file/d/1hlVTLZkveYGWpE9-46Wp5NVZt1slz-1T/view?usp=sharing)) without additional data on CIFAR10 under L_inf (8/255), run
```
python eval.py --arch WideResNet34 --checkpoint CKPT_DIR --data CIFAR10 --preprocess '01'
```
- For the SOTA result ([checkpoint](https://drive.google.com/file/d/1o8qQrYKQuHNKSH0kUBfFKruE1neN0h6W/view?usp=sharing)) without additional data on CIFAR100 under L_inf (8/255), run
```
python eval.py --arch WideResNet34 --checkpoint CKPT_DIR --data CIFAR100 --preprocess 'meanstd'
```
- For the SOTA result ([checkpoint](https://drive.google.com/file/d/1D-QCH-0ShtFo0s6gke5y6Ix7_x6k4Bys/view?usp=sharing)) without additional data on CIFAR10 under L_2 (0.5), run
```
python eval.py --arch WideResNet34 --checkpoint CKPT_DIR --data CIFAR10 --preprocess '01' --norm L2 --epsilon 0.5
```
