## Description of codes on ImageNet

Our experiments on ImageNet implement the FreeAT and FreeAT frameworks, where the codes are mainly forked from the [free adversarial training repository](https://github.com/mahyarnajibi/FreeAdversarialTraining), with the corresponding modifications with HE.

## Training codes

### FreeAT

Following the [FreeAT paper](https://arxiv.org/abs/1904.12843) by Shafahi et al. (2019), the command for training models with the FreeAT is
```shell
python main_free.py /data/ImageNet
```
The default training settings are included in the `configs.yml` file, where the model is `resnet50`, initial learning rate of momentum SGD is `0.1`, the maximal perturbation is `4/255` with step size of `1/255`, the batch size is `256`.

### FreeAT + HE

The training command for FreeAT + HE (our method) is
```shell
python main_free.py /data/ImageNet --FN True --WN True --s_HE 10.0 --angular_margin_HE 0.2
```
Here we activate the `FN` and `WN` operations, and set the scale `s=10.0` and margin `m=0.2`. These flags can be changed to perform, e.g., ablations studies on the effect of each component in HE.

## Evaluation codes

### Under the PGD attacks
To evaluate the performance of the trained models under the PGD attacks, the running command for *FreeAT* is
```shell
python evaluate.py /data/ImageNet -e --resume ./resnet50_free_adv_step4_eps4_repeat4_bs256/model_best.pth.tar
```
The running command for *FreeAT + HE* is
```shell
python evaluate.py /data/ImageNet -e --FN True --WN True --s_HE 10.0 --angular_margin_HE 0.2 --resume ./resnet50_free_adv_step4_eps4_repeat4_bs256_FN_WN_s10.0_margin0.2/model_best.pth.tar
```

### On the ImageNet-C dataset
To evaluate the performance of the general robustness, we also test on the [ImageNet-C](https://github.com/hendrycks/robustness) datasets by [Hendrycks & Dietterich (ICLR 2019)](https://arxiv.org/abs/1903.12261).
The running command for *FreeAT* is
```shell
python evaluate.py /data/ImageNet-C --eva_on_imagenet_c --resume ./resnet50_free_adv_step4_eps4_repeat4_bs256/model_best.pth.tar
```
and similarly the command for *FreeAT + HE* is 
```shell
python evaluate.py /data/ImageNet-C --eva_on_imagenet_c --FN True --WN True --s_HE 10.0 --angular_margin_HE 0.2 --resume ./resnet50_free_adv_step4_eps4_repeat4_bs256_FN_WN_s10.0_margin0.2/model_best.pth.tar
```

### Calculate gradient norm ratios
In order to investigate the gradient norm ratios as indicated in the Figure 6 of our paper, the command for *FreeAT + HE* is
```shell
python main_free.py /data/ImageNet --FN True --WN True --s_HE 10.0 --angular_margin_HE 0.2 --print_gradients True
```
For *FreeAT*, one can simply remove the related flags in the above command.
