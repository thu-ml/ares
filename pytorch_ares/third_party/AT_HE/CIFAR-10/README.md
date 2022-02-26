## Description of codes on CIFAR-10

To ensure that our experiments perform fair comparison with previous work, we mainly adopt the public codes from [TRADES](https://github.com/yaodongyu/TRADES), and make some modifications on them to run the trials.

## Adversarial Training 

* Train PGD-AT
```shell
python train_adv_cifar10.py --loss=pgd
```

### Adversarial Training with HE
Here we activate the `HE` operation and set the scale `s` and margin `m`.

* Train PGD-AT + HE:
```shell
python train_adv_cifar10.py --loss=pgd_he --s=10.0 --m=0.2 --wd 5e-4
```

* Train TRADES + HE:
```shell
python train_adv_cifar10.py --loss=trades_he --beta=6.0 --s=15.0 --m=0.2
```

* Train ALP + HE:
```shell
python train_adv_cifar10.py --loss=alp_he --beta=0.5 --s=20.0 --m=0.3
```

Our pre-trained models for CIFAR-10 WideResNets (in width factor 20 and 10) are available, including [PGD+HE-wide20](http://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/weights/model-wideres-pgdHE-wide20.pt) and [PGD+HE-wide10](http://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/weights/model-wideres-pgdHE-wide10.pt). We find that our HE mechanism is more suitable for the PGD-AT framework, since its principled formula is more aligned with our analysis.

## Robustness Evaluation

### Under the PGD attacks
To evaluate the performance of the trained models under the PGD attacks (default PGD-20), the running command is 
```shell
  $ python pgd_attack_cifar10.py
```

### Under AutoAttack
We also evaluate a reliable attack, named [AutoAttack](https://github.com/fra31/auto-attack), the running command is 
```shell
  $ python test_autoattack.py --model-path=$CHECKPOINT_PATH$ --use_FNandWN --widen_factor=20

```

### On the CIFAR-10-C dataset

To evaluate the performance of the general robustness, we also test on the [CIFAR-10-C](https://zenodo.org/record/2535967) datasets by [Hendrycks & Dietterich (ICLR 2019)](https://arxiv.org/abs/1903.12261).
The running command is
```shell
  $ python general_attack_cifar10.py
```
