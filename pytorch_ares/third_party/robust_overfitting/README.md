# Overfitting in adversarially robust deep learning
A repository which implements the experiments for exploring the phenomenon of robust overfitting, where robust performance on the test performance degradessignificantly over training. Created by [Leslie Rice][leslie link], [Eric Wong][eric link], and [Zico Kolter][zico link]. See our paper on arXiv [here][arxiv]. 

## News
+ 04/10/2020 - The AutoAttack framework of [Croce & Hein (2020)][autoattack arxiv] evaluated our released models using this repository [here][autoattack]. On CIFAR10, our models trained with standard PGD and early stopping ranks at \#5 overall, and \#1 for defenses that do not rely on additional data. 
+ 02/26/2020 - arXiv posted and repository release

## Robust overfitting hurts - early stopping is essential! 
A large amount of research over the past couple years has looked into defending deep networks against adversarial examples, with significant improvements over the well-known PGD-based adversarial training defense. However, adversarial training doesn't always behave similarly to standard training. The main observation we find is that, unlike in standard training, training to convergence can significantly harm robust generalization, and actually increases robust test error well before training has converged, as seen in the following learning curve: 

![overfitting](https://github.com/locuslab/robust_overfitting/blob/master/cifar10_curve.png)

After the initial learning rate decay, the robust test error actually increases! As a result, training to convergence is bad for adversarial training, and oftentimes, simply training for one epoch after decaying the learning rate achieves the best robust error on the test set. This behavior is reflected across multiple datasets, different approaches to adversarial training, and both L-infinity and L-2 threat models. 

## No algorithmic improvements over PGD-based adversarial training 
We can apply this knowledge to PGD-based adversarial training (e.g. as done by the original paper [here](https://arxiv.org/abs/1706.06083)), and find that early stopping can substantially improve the robust test error by 8%! As a result, we find that PGD-based adversarial training is as good as existing SOTA methods for adversarial robustness (e.g. on par with or slightly better than [TRADES](https://github.com/yaodongyu/TRADES)). On the flipside, we note that the results reported by TRADES also rely on early stopping, as training the TRADES approach to convergence results in a significant increase in robust test error. Unfortunately, this means that all of the algorithmic gains over PGD in adversarially robust training can be equivalent obtained with early stopping. 

## What is in this repository? 
+ The experiments for CIFAR-10, CIFAR-100, and SVHN are in `train_cifar.py`, `train_cifar100.py`, `train_svhn.py` respectively. 
+ CIFAR-10 training with semisupervised data is done in `train_cifar_semisupervised_half.py`, and uses the 500K pseudo-labeled TinyImages data from <https://github.com/yaircarmon/semisup-adv>
+ TRADES training is done with the repository located at <https://github.com/yaodongyu/TRADES>, with the only modification being the changes to the learning rate schedule to train to convergence (to decay at epochs 100 and 150 out of 200 total epochs). 
+ For ImageNet training, we used the repository located at <https://github.com/MadryLab/robustness> with no modifications. The resulting logged data is stored in `.pth` files which can be loaded with `torch.load()` and are simply dictionaries of logged data. The scripts containing the parameters for resuming the ImageNet experiments can be found in `imagenet_scripts/`. 
+ Training logs are all located in the `experiments` folder, and each subfolder corresponds to a set of experiments carried in the paper. 

Model weights for the following models can be found in this [drive folder][model weights]:
+ The best checkpoints for CIFAR-10 WideResNets defined in `wideresnet.py` (in for width factor 10 and 20 (from the double descent curve trained against L-infinity)
+ The best checkpoints for SVHN / CIFAR-10 (L2) / CIFAR-100 / ImageNet models reported in Table 1 (the ImageNet checkpoints are in the format directly used by <https://github.com/MadryLab/robustness>). The remaining models are for the Preactivation ResNet18 defined in `preactresnet.py`. 

[leslie link]: https://leslierice1.github.io/
[eric link]: https://riceric22.github.io/
[zico link]: http://zicokolter.com/

[arxiv]: https://arxiv.org/abs/2002.11569
[model weights]: https://drive.google.com/drive/folders/110JHo_yH9zwIf1b12jKoG6dRonrow9eA?usp=sharing
[autoattack]: https://github.com/fra31/auto-attack
[autoattack arxiv]: https://arxiv.org/abs/2003.01690
