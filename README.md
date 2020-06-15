# RealSafe

__RealSafe__ is a Python library for adversarial machine learning research focusing on benchmarking adversarial robustness on image classification correctly and efficiently.

Feature overview:

- Built on TensorFlow, and support TensorFlow & PyTorch models with the same interface.
- Support [many attacks](https://github.com/thu-ml/realsafe/tree/master/realsafe/attack).
- Provide ready-to-use pre-trained baseline models ([8 on ImageNet](https://github.com/thu-ml/realsafe/tree/master/example/imagenet) & [8 on CIFAR10](https://github.com/thu-ml/realsafe/tree/master/example/cifar10)).
- Provide efficient & easy-to-use tools for benchmarking models.

## Installation

Since RealSafe is still under development, please clone the repository and install the package:

``` shell
git clone https://github.com/thu-ml/realsafe
cd realsafe/
pip install -e .
```

The `requirements.txt` includes its dependencies, you might want to change PyTorch's version as well as TensorFlow 1's version. TensorFlow 1.13 or later should work fine.

As for python version, Python 3.5 or later should work fine.

The Boundary attack and the Evolutionary attack require `mpi4py` and a working MPI with enough __localhost slots__. For example, you could set the `OMPI_MCA_rmaps_base_oversubscribe` environment variable to `yes` for OpenMPI.

## Download Datasets & Model Checkpoints

By default, RealSafe would save datasets and model checkpoints under the `~/.realsafe` directory. You could override it by setting the `REALSAFE_RES_DIR` environment variable to an alternative location.

We support 2 dataset. To download the CIFAR-10 dataset, please run:

``` shell
python3 realsafe/dataset/cifar10.py
```

To download the ImageNet dataset, please run:

``` shell
python3 realsafe/dataset/imagenet.py
```

for instructions.

RealSafe include third party models' code in the `third_party/` directory as git submodules. Before you use these models, you need to initialize these submodules:

``` shell
git submodule init
git submodule update --depth 1
```

The `example/cifar10` directory and `example/imagenet` directory include wrappers for these models. Run the model's `.py` file to download its checkpoint or view instructions for downloading. For example, if you want to download the ResNet56 model's checkpoint, please run:

``` shell
python3 example/cifar10/resnet56.py
```

## Documentation

We provide __API docs__ as well as __tutorials__ at https://realsafe.readthedocs.io/.

## Quick Examples

RealSafe provides command line interface to run benchmarks. For example, to run distortion benchmark on ResNet56 model for CIFAR-10 dataset using CLI:

```shell
python3 -m realsafe.benchmark.distortion_cli --method mim --dataset cifar10 --offset 0 --count 1000 --output mim.npy example/cifar10/resnet56.py --distortion 0.1 --goal ut --distance-metric l_inf --batch-size 100 --iteration 10 --decay-factor 1.0 --logger
```

This command would find the minimal adversarial distortion achieved using the MIM attack with decay factor of 1.0 on the `example/cifar10/resnet56.py` model with L∞ distance and save the result to `mim.npy`.

For more examples and usages (e.g. how to define new models), please browse our documentation website mentioned before.

## Citing RealSafe

If you find RealSafe useful, you could cite our paper on benchmarking adversarial robustness using all models, all attacks & defenses supported in RealSafe. We provide a BibTeX entry of this paper below:

```
@InProceedings{Dong_2020_CVPR,
  author = {Dong, Yinpeng and Fu, Qi-An and Yang, Xiao and Pang, Tianyu and Su, Hang and Xiao, Zihao and Zhu, Jun},
  title = {Benchmarking Adversarial Robustness on Image Classification},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```

