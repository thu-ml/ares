# Bag of Tricks for Adversarial Training
Empirical tricks for training state-of-the-art robust models on CIFAR-10. A playground for fine-tuning the basic adversarial training settings. 

[Bag of Tricks for Adversarial Training](https://openreview.net/forum?id=Xb8xvrtB8Ce) (ICLR 2021)

[Tianyu Pang](http://ml.cs.tsinghua.edu.cn/~tianyu/), [Xiao Yang](https://github.com/ShawnXYang), [Yinpeng Dong](http://ml.cs.tsinghua.edu.cn/~yinpeng/), [Hang Su](http://www.suhangss.me/), and [Jun Zhu](http://ml.cs.tsinghua.edu.cn/~jun/index.shtml).

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 18.04.4
- GPU: Geforce 2080 Ti or Tesla P100
- Cuda: 10.1, Cudnn: v7.6
- Python: 3.6
- PyTorch: >= 1.4.0
- Torchvision: >= 0.4.0

## Acknowledgement
The codes are modifed based on [Rice et al. 2020](https://github.com/locuslab/robust_overfitting), and the model architectures are implemented by [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).

## Threat Model
We consider the most widely studied setting:
- **L-inf norm constraint with the maximal epsilon be 8/255 on CIFAR-10**.
- **No accessibility to additional data, neither labeled nor unlabeled**.
- **Utilize the PGD-AT framework in [Madry et al. 2018](https://arxiv.org/abs/1706.06083)**.

(Implementations on the TRADES framework can be found [here](https://github.com/ShawnXYang/AT_HE))

## Trick Candidates
Importance rate: ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) *Critical*  ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) *Useful*  ![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) *Insignificance*

- **Early stopping w.r.t. training epochs** (![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) *Critical*).
Early stopping w.r.t. training epochs was first introduced in the [code of TRADES](https://github.com/yaodongyu/TRADES), and was later thoroughly studied by [Rice et al., 2020](https://arxiv.org/abs/2002.11569). Due to its effectiveness, we regard this trick as a default choice.

- **Early stopping w.r.t. attack intensity** (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) *Useful*). Early stopping w.r.t. attack iterations was studied by [Wang et al. 2019](proceedings.mlr.press/v97/wang19i/wang19i.pdf) and [Zhang et al. 2020](https://arxiv.org/abs/2002.11242). Here we exploit the strategy of the later one, where the authors show that this trick can promote clean accuracy. The relevant flags include `--earlystopPGD` indicates whether apply this trick, while '--earlystopPGDepoch1' and '--earlystopPGDepoch2' separately indicate the epoch to increase the tolerence t by one, as detailed in [Zhang et al. 2020](https://arxiv.org/abs/2002.11242). (*Note that early stopping attack intensity may degrade worst-case robustness under strong attacks*)

- **Warmup w.r.t. learning rate** (![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) *Insignificance*). Warmup w.r.t. learning rate was found useful for [FastAT](https://arxiv.org/abs/2001.03994), while [Rice et al., 2020](https://arxiv.org/abs/2002.11569) found that piecewise decay schedule is more compatible with early stop w.r.t. training epochs. The relevant flags include `--warmup_lr` indicates whether apply this trick, while `--warmup_lr_epoch` indicates the end epoch of the gradually increase of learning rate.

- **Warmup w.r.t. epsilon** (![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) *Insignificance*). [Qin et al. 2019](https://arxiv.org/abs/1907.02610) use warmup w.r.t. epsilon in their implementation, where the epsilon gradually increase from 0 to 8/255 in the first 15 epochs. Similarly, the relevant flags include `--warmup_eps` indicates whether apply this trick, while `--warmup_eps_epoch` indicates the end epoch of the gradually increase of epsilon.

- **Batch size** (![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) *Insignificance*). The typical batch size used for CIFAR-10 is 128 in the adversarial setting. In the meanwhile, [Xie et al. 2019](https://arxiv.org/pdf/1812.03411.pdf) apply a large batch size of 4096 to perform adversarial training on ImageNet, where the model is distributed on 128 GPUs and has quite robust performance. The relevant flag is `--batch-size`. According to [Goyal et al. 2017](https://arxiv.org/abs/1706.02677), we take bs=128 and lr=0.1 as a basis, and scale the lr when we use larger batch size, e.g., bs=256 and lr=0.2.

- **Label smoothing** (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) *Useful*). Label smoothing is advocated by [Shafahi et al. 2019](https://arxiv.org/abs/1910.11585) to mimic the adversarial training procedure. The relevant flags include `--labelsmooth` indicates whether apply this trick, while `--labelsmoothvalue` indicates the degree of smoothing applied on the label vectors. When `--labelsmoothvalue=0`, there is no label smoothing applied. (*Note that only moderate label smoothing (~0.2) is helpful, while exccessive label smoothing (>0.3) could be harmful, as observed in [Jiang et al. 2020](https://arxiv.org/abs/2006.13726)*)

- **Optimizer** (![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) *Insignificance*). Most of the AT methods apply SGD with momentum as the optimizer. In other cases, [Carmon et al. 2019](https://arxiv.org/abs/1905.13736) apply SGD with Nesterov, and [Rice et al., 2020](https://arxiv.org/abs/2002.11569) apply Adam for cyclic learning rate schedule. The relevant flag is `--optimizer`, which include common optimizers implemented by official Pytorch API and recently proposed gradient centralization trick by [Yong et al. 2020](https://arxiv.org/abs/2004.01461).

- **Weight decay** (![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) *Critical*). The values of weight decay used in previous AT methods mainly fall into `1e-4` (e.g., [Wang et al. 2019](proceedings.mlr.press/v97/wang19i/wang19i.pdf)), `2e-4` (e.g., [Madry et al. 2018](https://arxiv.org/abs/1706.06083)), and `5e-4` (e.g., [Rice et al., 2020](https://arxiv.org/abs/2002.11569)). We find that slightly different values of weight decay could largely affect the robustness of the adversarially trained models.

- **Activation function** (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) *Useful*). As shown in [Xie et al., 2020a](https://arxiv.org/pdf/2006.14536.pdf), the smooth alternatives of `ReLU`, including `Softplus` and `GELU` can promote the performance of adversarial training. The relevant flags are `--activation` to choose the activation, and `--softplus_beta` to set the beta for Softplus. Other hyperparameters are used by default in the code.

- **BN mode** (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) *Useful*). TRADES applies eval mode of BN when crafting adversarial examples during training, while PGD-AT methods implemented by [Madry et al. 2018](https://arxiv.org/abs/1706.06083) or [Rice et al., 2020](https://arxiv.org/abs/2002.11569) use train mode of BN to craft training adversarial examples. As indicated by [Xie et al., 2020b](https://arxiv.org/pdf/1906.03787.pdf), properly dealing with BN layers is critical to obtain a well-performed adversarially trained model, while train mode of BN during multi-step PGD process may blur the distribution. 


## Baseline setting (on CIFAR-10)
- **Architecture**: WideResNet-34-10
- **Optimizer**: Momentum SGD with default hyperparameters
- **Total epoch**: `110`
- **Batch size**: `128`
- **Weight decay**: `5e-4`
- **Learning rate**: `lr=0.1`; decay to `lr=0.01` at 100 epoch; decay to `0.001` at 105 epoch
- **BN mode**: eval

running command for training:
```python
python train_cifar.py --model WideResNet --attack pgd \
                      --lr-schedule piecewise --norm l_inf --epsilon 8 \
                      --epochs 110 --attack-iters 10 --pgd-alpha 2 \
                      --fname auto \
		      --optimizer 'momentum' \
		      --weight_decay 5e-4
                      --batch-size 128 \
		      --BNeval \
```

## Empirical Evaluations
*The evaluation results on the baselines are quoted from  [AutoAttack](https://arxiv.org/abs/2003.01690) ([evaluation code](https://github.com/P2333/Bag-of-Tricks-for-AT/blob/master/eval_cifar.py))*. 

Note that **OURS (TRADES)** below only change the weight decay value from `2e-4` (used in original TRADES) to `5e-4`, and train for 110 epochs (lr decays at 100 and 105 epochs). To run the evaluation script `eval_cifar.py`, the command should be
```python
python eval_cifar.py --out-dir 'path_to_the_model' --ATmethods 'TRADES'
```
Here `ATmethods` refer to the AT framework (e.g., PGDAT or TRADES).

### CIFAR-10 (eps = 8/255)
|paper           | Architecture | clean         | AA |
|---|:---:|:---:|:---:|
| **OURS (TRADES)**[[Checkpoint](http://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/bag_of_tricks/wide20_trades_eps8_tricks.pt)] | WRN-34-20| 86.43 | 54.39 |
| **OURS (TRADES)**[[Checkpoint](http://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/bag_of_tricks/wide10_trades_eps8_tricks.pt)] | WRN-34-10| 85.48 | 53.80 |
| [(Pang et al., 2020)](https://arxiv.org/abs/2002.08619) | WRN-34-20| 85.14 | 53.74 |
| [(Zhang et al., 2020)](https://arxiv.org/abs/2002.11242)| WRN-34-10| 84.52 | 53.51 |
| [(Rice et al., 2020)](https://arxiv.org/abs/2002.11569) | WRN-34-20| 85.34 | 53.35 |


### CIFAR-10 (eps = 0.031)
|paper           | Architecture | clean         | AA |
|---|:---:|:---:|:---:|
| **OURS (TRADES)**[[Checkpoint](http://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/bag_of_tricks/wide10_trades_tricks.pt)] | WRN-34-10| 85.34 | 54.64 |
| [(Huang et al., 2020)](https://arxiv.org/abs/2002.10319) | WRN-34-10| 83.48 | 53.34 |
| [(Zhang et al., 2019)](https://arxiv.org/abs/1901.08573) | WRN-34-10| 84.92 | 53.04 |
