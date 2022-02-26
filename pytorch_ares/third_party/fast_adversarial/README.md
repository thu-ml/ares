# Fast adversarial training using FGSM

*A repository that implements the fast adversarial training code using an FGSM adversary, capable of training a robust CIFAR10 classifier in 6 minutes and a robust ImageNet classifier in 12 hours. Created by [Eric Wong](https://riceric22.github.io), [Leslie Rice](https://leslierice1.github.io/), and [Zico Kolter](http://zicokolter.com). See our paper on arXiv [here][paper], which was inspired by the free adversarial training paper [here][freepaper] by Shafahi et al. (2019).*

[paper]: https://arxiv.org/abs/2001.03994
[freepaper]: https://arxiv.org/abs/1904.12843

## News
+ 12/19/2019 - Accepted to ICLR 2020
+ 1/14/2019 - arXiv posted and repository release

## What is in this repository? 
+ An implementation of the FGSM adversarial training method with randomized initialization for MNIST, CIFAR10, and ImageNet
+ [Cyclic learning rates](https://arxiv.org/abs/1506.01186) and mixed precision training using the [apex](https://nvidia.github.io/apex/) library to achieve DAWNBench-like speedups 
+ Pre-trained models using this code base
+ The ImageNet code is mostly forked from the [free adversarial training repository](https://github.com/mahyarnajibi/FreeAdversarialTraining), with the corresponding modifications for fast FGSM adversarial training

## Installation and usage
+ All examples can be run without mixed-precision with PyTorch v1.0 or higher
+ To use mixed-precision training, follow the apex installation instructions [here](https://github.com/NVIDIA/apex#quick-start)

## But wait, I thought FGSM training didn't work!
As one of the earliest methods for generating adversarial examples, the Fast Gradient Sign Method (FGSM) is also known to be one of the weakest. It has largely been replaced by the PGD-based attacked, and it's use as an attack has become highly discouraged when [evaluating adversarial robustness](https://arxiv.org/abs/1902.06705). Afterall, early attempts at using FGSM adversarial training (including variants of randomized FGSM) were unsuccessful, and this was largely attributed to the weakness of the attack. 

However, we discovered that a fairly minor modification to the random initialization for FGSM adversarial training allows it to perform as well as the much more expensive PGD adversarial training. This was quite surprising to us, and suggests that one does not need very strong adversaries to learn robust models! As a result, we pushed the FGSM adversarial training to the limit, and found that by incorporating various techniques for fast training used in the [DAWNBench](https://dawn.cs.stanford.edu/benchmark/) competition, we could learn robust architectures an order of magnitude faster than before, while achieving the same degrees of robustness. A couple of the results from the paper are highlighted in the table below. 

|          | CIFAR10 Acc | CIFAR10 Adv Acc (eps=8/255) | Time (minutes) | 
| --------:| -----------:|----------------------------:|---------------:| 
| FGSM     |      86.06% |                      46.06% |             12 |
| Free     |      85.96% |                      46.33% |            785 |
| PGD      |      87.30% |                      45.80% |           4966 |

|          | ImageNet Acc | ImageNet Adv Acc (eps=2/255) | Time (hours) | 
| --------:| ------------:|-----------------------------:|-------------:| 
| FGSM     |       60.90% |                       43.46% |           12 |
| Free     |       64.37% |                       43.31% |           52 |

## But I've tried FGSM adversarial training before, and it didn't work! 
In our experiments, we discovered several failure modes which would cause FGSM adversarial training to ``catastrophically fail'', like in the following plot. 

![overfitting](https://github.com/locuslab/fast_adversarial/blob/master/overfitting_error_curve.png)

If FGSM adversarial training hasn't worked for you in the past, then it may be because of one of the following reasons (which we present as a non-exhaustive list of ways to fail): 

+ FGSM step size is too large, forcing the adversarial examples to cluster near the boundary
+ Random initialization only covers a smaller subset of the threat model
+ Long training with many epochs and fine tuning with very small learning rates

All of these pitfalls can be avoided by simply using early stopping based on a subset of the training data to evaluate the robust accuracy with respect to PGD, as the failure mode for FGSM adversarial training occurs quite rapidly (going to 0% robust accuracy within the span of a couple epochs)

## Why does this matter if I still want to use PGD adversarial training in my experiments? 

The speedups gained from using mixed-precision arithmetic and cyclic learning rates can still be reaped regardless of what training regimen you end up using! For example, these techniques can speed up CIFAR10 PGD adversarial training by almost 2 orders of magnitude, reducing training time by about 3.5 days to just over 1 hour. The engineering costs of installing the `apex` library and changing the learning rate schedule are miniscule in comparison to the time saved from using these two techniques, and so even if you don't use FGSM adversarial training, you can still benefit from faster experimentation with the DAWNBench improvements. 
