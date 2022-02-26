# HYDRA: Pruning Adversarially Robust Neural Networks (NeurIPS 2020)

Repository with code to reproduce the results and checkpoints for compressed networks in [our paper on novel pruning techniques with robust training](https://arxiv.org/abs/2002.10509). This repository supports all four robust training objectives: iterative adversarial training, randomized smoothing, MixTrain, and CROWN-IBP.

Following is a snippet of key results where we showed that accounting the robust training objective in pruning strategy can lead to large gains in the robustness of pruned networks. 

![results_table](/images/results_table.png)



In particular, the improvement arises from letting the robust training objective controlling which connections to prune. In almost all cases, it prefers to pruned certain high-magnitude weights while preserving other small magnitude weights, which is orthogonal to the strategy in well-established least-weight magnitude (LWM) based pruning. 

![weight_histogram](/images/weight_histogram.png)

## Updates
**April 30, 2020**: [Checkpoints for WRN-28-10](https://www.dropbox.com/sh/56yyfy16elwbnr8/AADmr7bXgFkrNdoHjKWwIFKqa?dl=0), a common network for benchmarking adv. robustness | 90% pruned with proposed technique | Benign test accuracy = 88.97% , PGD-50 test accuracy = 62.24%. 

**May 23, 2020**: Our WRN-28-10 network with 90% connection pruning comes in the second place in the [auto-attack robustness benchmark](https://github.com/fra31/auto-attack). 

## Getting started

Let's start by installing all dependencies. 

`pip install -r requirement.txt`



We will use `train.py` for all our experiments on the CIFAR-10 and SVHN dataset. For ImageNet, we will use `train_imagenet.py`. It provides the flexibility to work with pre-training, pruning, and Finetuning steps along with different training objectives.

- `exp_mode`: select from pretrain, prune, finetune
- `trainer`: benign (base), iterative adversarial training (adv), randomized smoothing (smooth), mix train, crown-imp 
- `--dataset`: cifar10, svhn, imagenet



Following [this](https://github.com/allenai/hidden-networks) work, we modify the convolution layer to have an internal mask. We can use a masked convolution layer with `--layer-type=subnet`. The argument `k` refers to the fraction of non-pruned connections.



## Pre-training

In pre-training, we train the networks with `k=1` i.e, without pruning. Following example pre-train a WRN-28-4 network with adversarial training.

`python train.py --arch wrn_28_4 --exp-mode pretrain --configs configs/configs.yml --trainer adv --val_method adv --k 1.0`



## Pruning

In pruning steps, we will essentially freeze weights of the network and only update the importance scores. The following command will prune the pre-trained WRN-28-4 network to 99% pruning ratio.  

`python train.py --arch wrn_28_4 --exp-mode prune --configs configs.yml --trainer adv --val_method adv --k 0.01 --scaled-score-init --source-net pretrained_net_checkpoint_path --epochs 20 --save-dense`

It used 20 epochs to optimize for better-pruned networks following the proposed scaled initialization of importance scores. It also saves a checkpoint of pruned with dense layers i.e, throws aways masks form each layer after multiplying it with weights. These dense checkpoints are helpful as they are directly loaded in a model based on standard layers from torch.nn. 



## Fine-tuning

In the fine-tuning step, we will update the non-pruned weights but freeze the importance scores. For correct results, we must select the same pruning ratio as the pruning step. 

`python train.py --arch wrn_28_4 --exp-mode finetune --configs configs.yml --trainer adv --val_method adv --k 0.01 --source-net pruned_net_checkpoint_path --save-dense --lr 0.01`



## Least weight magnitude (LWM) based pruning 

We use a single shot pruning approach where we prune the desired number of connections after pre-training in a single step. After that, the network is fine-tuned with a similar configuration as above. 

`python train.py --arch wrn_28_4 --exp-mode finetune --configs configs.yml --trainer adv --val_method adv --k 0.01 --source-net pretrained_net_checkpoint_path --save-dense --lr 0.01 --scaled-score-init`

The only difference from fine-tuning from previous steps is the now we initialized the importance scores with proposed scaling. This scheme effectively prunes the connection with the lowest magnitude at the start. Since the importance scores are not updated with fine-tuning, this will effectively implement the LWM based pruning. 



## Bringing it all together

We can use the following scripts to obtain compact networks from both LWM and proposed pruning techniques. 

- `get_compact_net_adv_train.sh`: Compact networks with iterative adversarial training. 
- `get_compact_net_rand_smoothing.sh` Compact networks with randomized smoothing.
- `get_compact_net_mixtrain.sh` Compact networks with MixTrain. 
- `get_compact_net_crown-ibp.sh` Compact networks with CROWN-IBP.





## Finding robust sub-networks

It is curious to ask whether pruning certain connections itself can induce robustness in a network. In particular, given a non-robust network, does there exist a highly robust subnetwork? We find that indeed there exist such robust subnetworks with a non-trivial amount of robustness. Here is an example to reproduce these results:

`python train.py --arch wrn_28_4 --configs configs.yml --trainer adv --val-method adv --k 0.5 --source-net pretrained_non-robust-net_checkpoint_path`

Thus, given the checkpoint path of a non-robust network, it aims to find a sub-network with half the connections but having high empirical robust accuracy. We can similarly optimize for verifiably robust accuracy by selecting `--trainer` from `smooth | mixtrain | crown-ibp`, with using proper configs for each. 



## Model Zoo (checkpoints for pre-trained and compressed networks)

We are releasing pruned models for all three pruning ratios (90, 95, 99%) for all three datasets used in the paper. In case you want to compare some additional property of pruned models with a baseline, we are also releasing non-pruned i.e., pre-trained networks. Note that, we use input normalization only for the ImageNet dataset. For each model, we are releasing two checkpoints: one with masked layers and other with dense layers. Note that the numbers from these checkpoints might differ a little bit from the ones reported in the paper.

### Adversarial training  

| Dataset | Architecture |                       Pre-trained (0%)                       |                          90% pruned                          |                          95% pruned                          |                          99% pruned                          |
| :-----: | :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| CIFAR10 |    VGG16     | [ckpt](https://www.dropbox.com/sh/1037dxc9m4m6wqs/AAD62kuJRuVaoRFOto_jxKJ2a?dl=0) | [ckpt](https://www.dropbox.com/sh/ugf2xokml5uf9s0/AAALs9dvG5fwejfBFU-RbL0ma?dl=0) | [ckpt](https://www.dropbox.com/sh/xehsrmls76k85y0/AAC-QARNd_b4hJYC5V9QwEJXa?dl=0) | [ckpt](https://www.dropbox.com/sh/8zgknaiv8o19o9k/AAAG2ZncZmhdj-Hz9uM46u-ka?dl=0) |
| CIFAR10 |   WRN-28-4   | [ckpt](https://www.dropbox.com/sh/zvqgjd5xx06lh3t/AACT5vYS3S6b33-0uRDjK2Awa?dl=0) | [ckpt](https://www.dropbox.com/sh/b9cyx9ewg5dt981/AADMA9vVVCXe68RwrSZtC9tia?dl=0) | [ckpt](https://www.dropbox.com/sh/cbt8xqq9na4tj1b/AADyPq6J34cUWHB8GvGf_ivDa?dl=0) | [ckpt](https://www.dropbox.com/sh/pjn8thd1fw2pujr/AABcCAH7BEdVrJs0v_gMQ0lTa?dl=0) |
|  SVHN   |    VGG16     | [ckpt](https://www.dropbox.com/sh/jmo7hj25po0r7tl/AAAw756-U1bifArFr_y1GeSCa?dl=0) | [ckpt](https://www.dropbox.com/sh/7pg0aaguyzndx61/AABqL_8-XFhilpywT9jMHCHqa?dl=0) | [ckpt](https://www.dropbox.com/sh/m3t33ku6aqecv4u/AACykFCWN1-QwbMftvk-a-8na?dl=0) | [ckpt](https://www.dropbox.com/sh/d8il3fpzxvx4uhq/AACZF5GVuV5yzc781Ge5kkD9a?dl=0) |
|  SVHN   |   WRN-28-4   | [ckpt](https://www.dropbox.com/sh/0o906gxijsk4ruh/AAAAj-mwEnv7uNgildkeMqC-a?dl=0) | [ckpt](https://www.dropbox.com/sh/9hyh3iwnrjwvgon/AAC2a6vZSrN3DvzVaPeBhQ6Ya?dl=0) | [ckpt](https://www.dropbox.com/sh/5hs67w8yh9crhyx/AAB8Q4u_EE9rDlYkTF-bT95Ta?dl=0) | [ckpt](https://www.dropbox.com/sh/l0c1houep3w61b6/AAB9CXmKnOpmLe_VKkwB4Ovaa?dl=0) |



### Randomized smoothing

| Dataset | Architecture |                       Pre-trained (0%)                       |                          90% pruned                          |                          95% pruned                          |                          99% pruned                          |
| :-----: | :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| CIFAR10 |    VGG16     | [ckpt](https://www.dropbox.com/sh/y5n7000qt7004fu/AAC7eRNUkGQvFfoepwn6tTpaa?dl=0) | [ckpt](https://www.dropbox.com/sh/0pwxek9vom9cywl/AACDZ_-lmhsNK9BG1BlzWpLea?dl=0) | [ckpt](https://www.dropbox.com/sh/pe8mfstkxl621hb/AAAohk6M7o-NwRXUsvk-hLfCa?dl=0) | [ckpt](https://www.dropbox.com/sh/iahysjrj1dekzpw/AAAjvfsE9Xu1P_q23lAF7uNoa?dl=0) |
| CIFAR10 |   WRN-28-4   | [ckpt](https://www.dropbox.com/sh/4xwjxiyal1o7qr3/AABnCDX5dNin_NeYxmlS9XpLa?dl=0) | [ckpt](https://www.dropbox.com/sh/6jj33youpc41o4o/AAAfjYboGCg9yZc-XYyL3ABza?dl=0) | [ckpt](https://www.dropbox.com/sh/3qqw15yyza5zi6a/AABDVyGvJcCEyWT6kPDOQ-spa?dl=0) | [ckpt](https://www.dropbox.com/sh/m1dvdgedovb19yp/AACxxW-6xArpiVV4cfY7cwAYa?dl=0) |
|  SVHN   |    VGG16     | [ckpt](https://www.dropbox.com/sh/9k82top60lvngqb/AABAX9wJUBqGmF8akhoWrRA6a?dl=0) | [ckpt](https://www.dropbox.com/sh/7siuxmb6l9d1qt1/AADnA4m4-1k27eZCBkGyU6ena?dl=0) | [ckpt](https://www.dropbox.com/sh/j0eh9jyqpqurvl3/AAAS4awDRQhiyEnNEPNqwlg2a?dl=0) | [ckpt](https://www.dropbox.com/sh/3rnl9uea4cb44vs/AACaTNrTsp5JybLoCAGzid-4a?dl=0) |
|  SVHN   |   WRN-28-4   | [ckpt](https://www.dropbox.com/sh/m5he7uskva23sfr/AADUlbsXAxuROXFo7Bt2U8R6a?dl=0) | [ckpt](https://www.dropbox.com/sh/hzymmaem17pcr68/AADeFeEZJ4X2fo6WCiqfA1tFa?dl=0) | [ckpt](https://www.dropbox.com/sh/b8kqbkcsmxlhdt9/AABFYwwUHxj3-cnCgL3f0pota?dl=0) | [ckpt](https://www.dropbox.com/sh/g2z07aucy9tw4z8/AABJ1inIcVX2UFdD3e75vjMNa?dl=0) |



### Adversarial training on ImageNet (ResNet50)

|                       Pre-trained (0%)                       |                          95% pruned                          |                          99% pruned                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [ckpt](https://www.dropbox.com/sh/z9m0mp6jkdp0ovi/AACtN93nnlp-u48WOgeuzb8Ra?dl=0) | [ckpt](https://www.dropbox.com/sh/w003d06uga1ylu4/AADBY9zbz9dgGYi2Ir2ZyINAa?dl=0) | [ckpt](https://www.dropbox.com/sh/i9i1i50een62zae/AAAq-HNkEsYS8dEmQY3sU4ERa?dl=0) |



## Contributors

* [Vikash Sehwag](https://vsehwag.github.io/)
* [Shiqi Wang](https://www.cs.columbia.edu/~tcwangshiqi/)



Some of the code in this repository is based on the following amazing works.

* https://github.com/allenai/hidden-networks
* https://github.com/yaircarmon/semisup-adv
* https://github.com/locuslab/smoothing
* https://github.com/huanzhang12/CROWN-IBP
* https://github.com/tcwangshiqi-columbia/symbolic_interval



## Reference

If you find this work helpful, consider citing it. 
```
@article{sehwag2020hydra,
  title={Hydra: Pruning adversarially robust neural networks},
  author={Sehwag, Vikash and Wang, Shiqi and Mittal, Prateek and Jana, Suman},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
