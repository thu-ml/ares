# CIFAR10 Adversarial Examples Challenge

Recently, there has been much progress on adversarial *attacks* against neural networks, such as the [cleverhans](https://github.com/tensorflow/cleverhans) library and the code by [Carlini and Wagner](https://github.com/carlini/nn_robust_attacks).
We now complement these advances by proposing an *attack challenge* for the
[CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) which follows the
format of [our earlier MNIST challenge](https://github.com/MadryLab/mnist_challenge).
We have trained a robust network, and the objective is to find a set of adversarial examples on which this network achieves only a low accuracy.
To train an adversarially-robust network, we followed the approach from our recent paper:

**Towards Deep Learning Models Resistant to Adversarial Attacks** <br>
*Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu* <br>
https://arxiv.org/abs/1706.06083.

As part of the challenge, we release both the training code and the network architecture, but keep the network weights secret.
We invite any researcher to submit attacks against our model (see the detailed instructions below).
We will maintain a leaderboard of the best attacks for the next two months and then publish our secret network weights.

Analogously to our MNIST challenge, the goal of this challenge is to clarify the state-of-the-art for adversarial robustness on CIFAR10. Moreover, we hope that future work on defense mechanisms will adopt a similar challenge format in order to improve reproducibility and empirical comparisons.

**Update 2017-12-10**: We released our secret model. You can download it by running `python fetch_model.py secret`. As of Dec 10 we are no longer accepting black-box challenge submissions. We have set up a leaderboard for white-box attacks on the (now released) secret model. The submission format is the same as before. We plan to continue evaluating submissions and maintaining the leaderboard for the foreseeable future.

## Black-Box Leaderboard (Original Challenge)

| Attack                                 | Submitted by  | Accuracy | Submission Date |
| -------------------------------------- | ------------- | -------- | ---- |
| PGD on the cross-entropy loss for the<br> adversarially trained public network     | (initial entry)       | **63.39%**   | Jul 12, 2017    |
| PGD on the [CW](https://github.com/carlini/nn_robust_attacks) loss for the<br> adversarially trained public network     | (initial entry)       | 64.38%   | Jul 12, 2017    |
| FGSM on the [CW](https://github.com/carlini/nn_robust_attacks) loss for the<br> adversarially trained public network     | (initial entry)       | 67.25%   | Jul 12, 2017    |
| FGSM on the [CW](https://github.com/carlini/nn_robust_attacks) loss for the<br> naturally trained public network     | (initial entry)       | 85.23%   | Jul 12, 2017    |

## White-Box Leaderboard

| Attack                                 | Submitted by  | Accuracy | Submission Date |
| -------------------------------------- | ------------- | -------- | ---- |
| Guided Local Attack | Siyuan Yi | **43.95%**   | Aug 2, 2021    |
| [EWR-PGD](https://github.com/liuye6666/EWR-PGD) | Ye Liu | 43.96%   | Sep 8, 2020    |
| [PGD attack with Output Diversified Initialization](https://arxiv.org/abs/2003.06878) | Yusuke Tashiro | 43.99%   | Feb 15, 2020    |
| [MultiTargeted](https://arxiv.org/abs/1910.09338) | Sven Gowal | 44.03%   | Aug 28, 2019    |
| [FAB: Fast Adaptive Boundary Attack](https://github.com/fra31/fab-attack) | Francesco Croce       | 44.51%   | Jun 7, 2019    |
| [Distributionally Adversarial Attack](https://github.com/tianzheng4/Distributionally-Adversarial-Attack) | Tianhang Zheng       | 44.71%   | Aug 21, 2018    |
| 20-step PGD on the cross-entropy loss<br> with 10 random restarts | Tianhang Zheng       | 45.21%   | Aug 24, 2018    |
| 20-step PGD on the cross-entropy loss | (initial entry)       | 47.04%   | Dec 10, 2017    |
| 20-step PGD on the [CW](https://github.com/carlini/nn_robust_attacks) loss | (initial entry)       | 47.76%   | Dec 10, 2017    |
| FGSM on the [CW](https://github.com/carlini/nn_robust_attacks) loss | (initial entry)       | 54.92%   | Dec 10, 2017    |
| FGSM on the cross-entropy loss | (initial entry)       | 55.55%   | Dec 10, 2017    |





## Format and Rules

The objective of the challenge is to find black-box (transfer) attacks that are effective against our CIFAR10 model.
Attacks are allowed to perturb each pixel of the input image by at most `epsilon=8.0` on a `0-255` pixel scale.
To ensure that the attacks are indeed black-box, we release our training code and model architecture, but keep the actual network weights secret. 

We invite any interested researchers to submit attacks against our model.
The most successful attacks will be listed in the leaderboard above.
As a reference point, we have seeded the leaderboard with the results of some standard attacks.

### The CIFAR10 Model

We used the code published in this repository to produce an adversarially robust model for CIFAR10 classification. The model is a residual convolutional neural network consisting of five residual units and a fully connected layer. This architecture is derived from the "w32-10 wide" variant of the [Tensorflow model repository](https://github.com/tensorflow/models/blob/master/resnet/resnet_model.py).
The network was trained against an iterative adversary that is allowed to perturb each pixel by at most `epsilon=8.0`.

The random seed used for training and the trained network weights will be kept secret.

The `sha256()` digest of our model file is:
```
555be6e892372599380c9da5d5f9802f9cbd098be8a47d24d96937a002305fd4
```
We will release the corresponding model file on September 15 2017, which is roughly two months after the start of this competition. **Edit: We are extending the deadline for submitting attacks to October 15th due to requests.**

### The Attack Model

We are interested in adversarial inputs that are derived from the CIFAR10 test set.
Each pixel can be perturbed by at most `epsilon=8.0` from its initial value on the `0-255` pixel scale.
All pixels can be perturbed independently, so this is an l_infinity attack.

### Submitting an Attack

Each attack should consist of a perturbed version of the CIFAR10 test set.
Each perturbed image in this test set should follow the above attack model. 

The adversarial test set should be formated as a numpy array with one row per example and each row containing a 32x32x3
array of pixels.
Hence the overall dimensions are 10,000x32x32x3.
Each pixel must be in the [0, 255] range.
See the script `pgd_attack.py` for an attack that generates an adversarial test set in this format.

In order to submit your attack, save the matrix containing your adversarial examples with `numpy.save` and email the resulting file to cifar10.challenge@gmail.com. 
We will then run the `run_attack.py` script on your file to verify that the attack is valid and to evaluate the accuracy of our secret model on your examples.
After that, we will reply with the predictions of our model on each of your examples and the overall accuracy of our model on your evaluation set.

If the attack is valid and outperforms all current attacks in the leaderboard, it will appear at the top of the leaderboard.
Novel types of attacks might be included in the leaderboard even if they do not perform best.

We strongly encourage you to disclose your attack method.
We would be happy to add a link to your code in our leaderboard.

## Overview of the Code
The code consists of seven Python scripts and the file `config.json` that contains various parameter settings.

### Running the code
- `python train.py`: trains the network, storing checkpoints along
      the way.
- `python eval.py`: an infinite evaluation loop, processing each new
      checkpoint as it is created while logging summaries. It is intended
      to be run in parallel with the `train.py` script.
- `python pgd_attack.py`:  applies the attack to the CIFAR10 eval set and
      stores the resulting adversarial eval set in a `.npy` file. This file is
      in a valid attack format for our challenge.
- `python run_attack.py`: evaluates the model on the examples in
      the `.npy` file specified in config, while ensuring that the adversarial examples 
      are indeed a valid attack. The script also saves the network predictions in `pred.npy`.
- `python fetch_model.py name`: downloads the pre-trained model with the
      specified name (at the moment `adv_trained` or `natural`), prints the sha256
      hash, and places it in the models directory.
- `cifar10_input.py` provides utility functions and classes for loading the CIFAR10 dataset.

### Parameters in `config.json`

Model configuration:
- `model_dir`: contains the path to the directory of the currently 
      trained/evaluated model.

Training configuration:
- `tf_random_seed`: the seed for the RNG used to initialize the network
      weights.
- `numpy_random_seed`: the seed for the RNG used to pass over the dataset in random order
- `max_num_training_steps`: the number of training steps.
- `num_output_steps`: the number of training steps between printing
      progress in standard output.
- `num_summary_steps`: the number of training steps between storing
      tensorboard summaries.
- `num_checkpoint_steps`: the number of training steps between storing
      model checkpoints.
- `training_batch_size`: the size of the training batch.

Evaluation configuration:
- `num_eval_examples`: the number of CIFAR10 examples to evaluate the
      model on.
- `eval_batch_size`: the size of the evaluation batches.
- `eval_on_cpu`: forces the `eval.py` script to run on the CPU so it does not compete with `train.py` for GPU resources.

Adversarial examples configuration:
- `epsilon`: the maximum allowed perturbation per pixel.
- `k`: the number of PGD iterations used by the adversary.
- `a`: the size of the PGD adversary steps.
- `random_start`: specifies whether the adversary will start iterating
      from the natural example or a random perturbation of it.
- `loss_func`: the loss function used to run pgd on. `xent` corresponds to the
      standard cross-entropy loss, `cw` corresponds to the loss function 
      of [Carlini and Wagner](https://arxiv.org/abs/1608.04644).
- `store_adv_path`: the file in which adversarial examples are stored.
      Relevant for the `pgd_attack.py` and `run_attack.py` scripts.

## Example usage
After cloning the repository you can either train a new network or evaluate/attack one of our pre-trained networks.
#### Training a new network
* Start training by running:
```
python train.py
```
* (Optional) Evaluation summaries can be logged by simultaneously
  running:
```
python eval.py
```
#### Download a pre-trained network
* For an adversarially trained network, run
```
python fetch_model.py adv_trained
```
and use the `config.json` file to set `"model_dir": "models/adv_trained"`.
* For a naturally trained network, run
```
python fetch_model.py natural
```
and use the `config.json` file to set `"model_dir": "models/naturally_trained"`.
#### Test the network
* Create an attack file by running
```
python pgd_attack.py
```
* Evaluate the network with
```
python run_attack.py
```
