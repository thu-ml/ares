# Feature Scattering Adversarial Training (NeurIPS 2019)

## Introduction
This is the implementation of the
["Feature-Scattering Adversarial Training"](https://papers.nips.cc/paper/8459-defense-against-adversarial-attacks-using-feature-scattering-based-adversarial-training.pdf), which is a training method for improving model robustness against adversarial attacks. It advocates the usage of an unsupervised feature-scattering procedure for adversarial perturbation generation, which is effective for overcoming label leaking and improving model robustness.
More information can be found on the project page: https://sites.google.com/site/hczhang1/projects/feature_scattering

## Usage
### Installation
The training environment (PyTorch and dependencies) can be installed as follows:
```
git clone https://github.com/Haichao-Zhang/FeatureScatter.git
cd FeatureScatter

python3 -m venv .venv
source .venv/bin/activate

python3 setup.py install

(or pip install -e .)
```
Tested under Python 3.5.2 and PyTorch 1.2.0.

### Train
Specify the path for saving the trained models in ```fs_train.sh```, and then run
```
sh ./fs_train.sh
```

### Evaluate
Specify the path to the trained models to be evaluated in ```fs_eval.sh``` and then run
```
sh ./fs_eval.sh
```

### Reference Model
A reference model trained on CIFAR10 is [here](https://drive.google.com/open?id=1FXgE7llvQoypf7iCGR680EKQf9cARTSg).


## Cite

If you find this work is useful, please cite the following:

```
@inproceedings{feature_scatter,
    author = {Haichao Zhang and Jianyu Wang},
    title  = {Defense Against Adversarial Attacks Using Feature Scattering-based Adversarial Training},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2019}
}
```

## Contact

For questions related to feature-scattering, please send me an email: ```hczhang1@gmail.com```
