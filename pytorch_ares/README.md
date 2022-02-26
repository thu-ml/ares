# pytorch_ares
This repository contains the code for ARES (Adversarial Robustness Evaluation for Safety), 
a Python library for adversarial machine learning research focusing on benchmarking adversarial 
robustness on image classification correctly and comprehensively.
## Installation

- Clone this repo
    ```bash
    git clone https://github.com/thu-ml/ares/tree/main/pytorch_ares
    ```
- Install the experimental environment
    ```bash
    pip install -r requirements.txt
    ```
The requirements.txt includes its dependencies.
## Files in the folder
- `pytorch_ares/`
  - `data/`: The code supports cifar10 and imagenet datasets.
  - `test/`: Some toyexamples for testing adversarial attack methods and adversarial defense methods.
  - `pytorch_ares/`
    - `dataset_torch/`: Data processing for cifar10 and imagenet datasets.
    - `attack_torch/`: PyTorch implementation of some adversarial attack methods.
    - `cifar10_model/`: PyTorch implementation of some adversarial defense models on the cifar10 dataset.
    - `defense_torch/`: PyTorch implementation of some defense methods.
  - `third_party/`: Other open source repositories.
  - `attack_benchmark/`: Adversarial robustness benchmarks for image classification.
 ## Supported Methods

### Adversarial attack

- **FGSM**: [Explaining and harnessing adversarial examples](https://arxiv.org/pdf/1412.6572.pdf)
- **BIM**: [Adversarial examples in the physical world](https://arxiv.org/pdf/1607.02533.pdf?ref=https://githubhelp.com)
- **PGD**: [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/pdf/1706.06083.pdf中有体现，以后说到CW攻击再细说%E3%80%82)
- **CW**: [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/pdf/1608.04644.pdf?source=post_page)
- **DeepFool**: [DeepFool: a simple and accurate method to fool deep neural networks](https://openaccess.thecvf.com/content_cvpr_2016/papers/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.pdf)
- **MIM**: [Boosting Adversarial Attacks with Momentum](https://openaccess.thecvf.com/content_cvpr_2018/papers/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.pdf)
- **DIM**: [Improving Transferability of Adversarial Examples with Input Diversity](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xie_Improving_Transferability_of_Adversarial_Examples_With_Input_Diversity_CVPR_2019_paper.pdf)
- **TIM**: [Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dong_Evading_Defenses_to_Transferable_Adversarial_Examples_by_Translation-Invariant_Attacks_CVPR_2019_paper.pdf)
- **SI-NI-FGSM**: [Nesterov accelerated gradient and scale invariance for adversarial attacks](https://arxiv.org/pdf/1908.06281.pdf)
- **VIM**: [Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Enhancing_the_Transferability_of_Adversarial_Attacks_Through_Variance_Tuning_CVPR_2021_paper.pdf)
- **SGM**: [Skip connections matter: On the transferability of adversarial examples generated with resnets](https://arxiv.org/pdf/2002.05990.pdf)
- **CDA**: [Cross-Domain Transferability of Adversarial Perturbations](https://proceedings.neurips.cc/paper/2019/file/99cd3843754d20ec3c5885d805db8a32-Paper.pdf)
- **AutoAttack**: [Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks](http://proceedings.mlr.press/v119/croce20b/croce20b.pdf)
- **Boundary**: [Decision-based adversarial attacks: Reliable attacks against black-box machine learning models](https://arxiv.org/pdf/1712.04248.pdf)
- **SPSA**: [Adversarial Risk and the Dangers of Evaluating Against Weak Attacks](http://proceedings.mlr.press/v80/uesato18a/uesato18a.pdf)
- **Evolutionary**: [Efficient Decision-based Black-box Adversarial Attacks on Face Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dong_Efficient_Decision-Based_Black-Box_Adversarial_Attacks_on_Face_Recognition_CVPR_2019_paper.pdf)
- **NES**: [Black-box Adversarial Attacks with Limited Queries and Information](http://proceedings.mlr.press/v80/ilyas18a/ilyas18a.pdf)
- **Nattack**: [NATTACK: Learning the Distributions of Adversarial Examples for an Improved Black-Box Attack on Deep Neural Networks](http://proceedings.mlr.press/v97/li19g/li19g.pdf)
- **TTA**: [On Success and Simplicity: A Second Look at Transferable Targeted Attacks](https://proceedings.neurips.cc/paper/2021/file/30d454f09b771b9f65e3eaf6e00fa7bd-Paper.pdf)

### Adversarial defense

- **RST**: [Unlabeled Data Improves Adversarial Robustness](https://arxiv.org/abs/1905.13736)

- **TRADES**: [Theoretically Principled Trade-off between Robustness and Accuracy](https://arxiv.org/abs/1901.08573)
- **FS-AT**: [Defense Against Adversarial Attacks Using Feature Scattering-based Adversarial Training](https://arxiv.org/abs/1907.10764)
- **Pre-Training**: [Using Pre-Training Can Improve Model Robustness and Uncertainty](https://arxiv.org/abs/1907.10764)
- **AT-HE**: [Boosting Adversarial Training with Hypersphere Embedding](https://arxiv.org/abs/2002.08619)
- **Robust Overfitting**: [Overfitting in adversarially robust deep learning](https://arxiv.org/abs/2002.11569)
- **FastAT**: [Fast is better than free: Revisiting adversarial training](https://arxiv.org/abs/2001.03994)
- **AWP**: [Adversarial Weight Perturbation Helps Robust Generalization](https://arxiv.org/abs/2004.05884)
- **HYDRA**: [HYDRA: Pruning Adversarially Robust Neural Networks](https://arxiv.org/abs/2002.10509)
- **Label Smoothing**: [Bag of Tricks for Adversarial Training](https://arxiv.org/abs/2010.00467)

## Example to run the codes

ARES provides command line interface to run benchmarks. For example, you can test the attack success rate of fgsm on resnet18 on the cifar10 dataset:

    cd test/
    python test_white_box_attack.py --attack_name fgsm --dataset_name cifar10

There are 4 run_***.py files in the attack_benchmark folder that evaluate the adversarial robustness benchmarks on the cifar10 and imagenet datasets. For example, if you want to evaluate the robustness of the defense model on the cifar10 dataset, you can run the following command line:

    cd attack_benchmark/ 
    python run_cifar10_defense.py  



