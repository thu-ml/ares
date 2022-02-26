# EasyRobust

EasyRobust is a tool for training your robust models. Now it support adversarial / non-adversarial training of CNN / ViT models on ImageNet.

## Install

```
git clone https://github.com/alibaba/easyrobust.git
cd easyrobust
pip install -e .
```

## Training

```
sh train.sh train_configs/imagenet/resnet50_baseline.yaml
```

## Testing

see [test_scripts](https://github.com/alibaba/easyrobust/tree/main/test_scripts)

## Templates for Training

- `train_configs/imagenet/deit_small_baseline.yaml`: baseline training on deit_small
- `train_configs/imagenet/resnet50_baseline.yaml`: baseline training on resnet50
- `train_configs/imagenet/advtrain/resnet50_advtrain.yaml`: adversarial training on resnet50
- `train_configs/imagenet/advtrain/deit_small_advtrain.yaml`: adversarial training on deit_small

More training templates will be supported in future.

## Supported Methods

### Augmentation-Based

- All data augmentation in timm.

- **StyleAugmentation** [Style Augmentation: Data Augmentation via Style Randomization](https://arxiv.org/abs/1809.05375)

- **CartoonAugmentation** [CartoonGAN: Generative Adversarial Networks for Photo Cartoonization](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)

### Model-Based
- All implemented models in timm.

- **WaveCNet**: [WaveCNet: Wavelet Integrated CNNs to Suppress Aliasing Effect for Noise-Robust Image Classification](https://arxiv.org/abs/2107.13335)

- **RVT**: [Towards Robust Vision Transformer](https://arxiv.org/abs/2105.07926)

- **DrViT**: [Discrete Representations Strengthen Vision Transformer Robustness](https://arxiv.org/abs/2111.10493)

### Activation-Based

- **kWTA**: [Resisting Adversarial Attacks by k-Winners-Take-All](https://arxiv.org/abs/1905.10510)

- **LP_ReLU**: [Robust Image Classification Using a Low-Pass Activation Function and DCT Augmentation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9455411&tag=1)

### Pooling-Based

- **BlurPool**: [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486)

- **GaussianPool**: [Gaussian-Based Pooling for Convolutional Neural Networks](https://staff.aist.go.jp/takumi.kobayashi/publication/2019/NeurIPS2019.pdf)

### Norm-Based

- **SelfNorm**: [CrossNorm and SelfNorm for Generalization under Distribution Shifts](https://arxiv.org/abs/2102.02811)

- **pAdaIN**: [Permuted AdaIN: Reducing the Bias Towards Global Statistics in Image Classification](https://arxiv.org/abs/2010.05785)

- **CrossNorm**: [CrossNorm and SelfNorm for Generalization under Distribution Shifts](https://arxiv.org/abs/2102.02811)

### Training Methods

- **Adversarial Training**: [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)

- **GNT**: [A simple way to make neural networks robust against diverse image corruptions](https://arxiv.org/abs/2001.06057)

- **Shape-Texture Debiased**: [Shape-Texture Debiased Neural Network Training](https://arxiv.org/abs/2010.05981)

- **AdvProp**: [Adversarial Examples Improve Image Recognition](https://arxiv.org/abs/1911.09665)

### Loss function

- **ProbCompactLoss**: [Improving Adversarial Robustness via Probabilistically Compact Loss with Logit Constraints](https://arxiv.org/abs/2012.07688)

- **CEB**: [CEB Improves Model Robustness](https://arxiv.org/abs/2002.05380)

## TODO

- More training scripts
- Validation the results
- More robust methods

