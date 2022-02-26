
# Unlabeled Data Improves Adversarial Robustness  
  
This repository contains code for reproducing data and models from the NeurIPS 2019 paper [Unlabeled Data Improves Adversarial Robustness](https://arxiv.org/pdf/1905.13736.pdf) by Yair Carmon, Aditi Raghunathan, Ludwig Schmidt, Percy Liang and John C. Duchi. 

## CIFAR-10 unlabeled data and trained models  

Below are links to files containing our unlabeled data and pretrained models:

- [500K unlabeled data from TinyImages (with pseudo-labels)](https://drive.google.com/open?id=1LTw3Sb5QoiCCN-6Y5PEKkq9C9W60w-Hi)
- [Trained *heuristic* defense RST_adv(50K+500K) model (see Table 1 in the paper)](https://drive.google.com/open?id=1S3in_jVYJ-YBe5-4D0N70R4bN82kP5U2)
- [Trained *certified* defense RST_stab(50K+500K) model (see Figure 1 in the paper)](https://drive.google.com/open?id=1qNCQf1S47W9DPurUN4SKakmU87wE7ZRv)  
  
Additional files:  
- [Trained data sourcing model  
 (Classifies between CIFAR-10 and non-CIFAR-10 content)](https://drive.google.com/open?id=1neK7UPhX7muJM7GvUtYSPZB3yan8iy5b) 
- [TinyImages indices with keywords matching CIFAR-10  
(from the CIFAR-10.1 paper)](https://drive.google.com/open?id=1OaAGYLxr62t7Zby6F0jScMORnadk6Oz2)
- [Nearest neighbor L2 distances between CIFAR-10 test set and TinyImages](https://drive.google.com/open?id=1yMDnCfByqE6Y3l44844zF4fzjTyXaeKs) 
  
## Dependencies  
To create a conda environment called semisup-adv containing all the dependencies, run  
```  
conda env create -f environment.yml  
```  
  
Note: We tested this code on 2 GPUs in parallel, each with 12GB of memory. Running on CPUs or GPUs with less memory might require adjustments.  
  
The code in this repo is based on code from the following sources:  
- TRADES: https://github.com/yaodongyu/TRADES  
- Randomized smoothing: https://github.com/locuslab/smoothing  
- AutoAugment: https://github.com/DeepVoltaire/AutoAugment  
- Cutout: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py  
- FoolBox: https://github.com/bethgelab/foolbox  
- ShakeShake: https://github.com/hysts/pytorch_shake_shake  
- CIFAR-10.1: https://github.com/modestyachts/CIFAR-10.1
  
## Running robust self-training  
To run robust self-training you will a pickle file containing pseudo-labeled data. You can download ```ti_500K_pseudo_labeled.pickle``` containing our 500K pseudo-labeled TinyImages, or you can generate one from scratch using the instructions below.  
  
### Adversarial training with TRADES  
The following command performs adversarial training and produces a model  
equivalent to  RST_adv(50K+500K) described in the paper.  
```  
python robust_self_training.py --aux_data_filename ti_500K_pseudo_labeled.pickle --distance l_inf --epsilon 0.031 --model_dir rst_adv 
```  
  
When the script finishes running there will a be checkpoint file called `rst_adv/checkpoint-epoch200.pt`. The following commands runs a PGD attack (PGD_Ours from the paper) on the model  
```  
python attack_evaluation.py --model_path rst_adv/checkpoint-epoch200.pt --attack pgd --output_suffix pgd_ours  
```  
  
To run the Carlini-Wanger attack on randomly selected 1000 images from the test set, use  
```  
python attack_evaluation.py --model_path rst_adv/checkpoint-epoch200.pt --attack cw --output_suffix cw --num_eval_batches 5 --shuffle_testset  
```  
  
### Stability training  
The following commands performs stability training and produces a model equivalent to  
RST_stab(50K+500K) described in the paper.  
```  
python robust_self_training.py --aux_data_filename ti_500K_pseudo_labeled.pickle --distance l_2 --epsilon 0.25 --model_dir rst_stab --epochs 800
 ```
 
When the script finishes running there will a be checkpoint file called `rst_stab/checkpoint-epoch800.pt`.  The following commands runs randomized smoothing certification on the model, as described in the paper.  
```  
python smoothing_evaluation.py --model_path rst_stab/checkpoint-epoch800.pt --sigma 0.25  
```
  
## Creating the unlabeled data from scratch  

Note: creating the unlabeled data from scratch takes a while; plan for three days at least.

### Step zero: Downloading data 
Create a data directory that has the following files: 
- [tiny_images.bin](http://horatio.cs.nyu.edu/mit/tiny/data/tiny_images.bin)
- [TinyImages keyword information](https://drive.google.com/open?id=1OaAGYLxr62t7Zby6F0jScMORnadk6Oz2)

### Step one: Tiny Image preliminaries  
In this step, we do the following two preliminary steps.   
1) Compute distances from all the TinyImages to CIFAR-10 test set, in order to ensure we do *not* add any images from the test set to the unlabeled data sourced from TinyImages.   
2) Create train/test data for selection model (See Appendix B.6)  
    
Note that the data directory should contain the following files: `tiny_images.bin`, `cifar10_keywords_unique_v7.json`, `tinyimage_subset_indices_v7.json`  and `tinyimage_subset_data_v7.pickle`.
  
Here is an example run.  
 ``` 
 python tinyimages_preliminaries.py --data_dir ../data/ --output_dir ../data
 ```  
  
### Step two: Train a selection model
Here we train the data selection model described in Appendix B.6 of the paper.  Note that `data_dir` should contain the following files: `tiny_images.bin`, `ti_vs_cifar_inds.pickle` (from above).   
  
Here is an example run.   

 ```
 python train_cifar10_vs_ti.py --output_dir ../cifar10-vs-ti/ --data_dir ../data/  
```  
  
  
### Step three:  Selecting unlabeled data and removing CIFAR-10 test set   
We apply the model trained above on TinyImages and select images based on the predictions, while making sure to remove all images that are close (in l2 distance) to the CIFAR-10 test set.   

 ```
python tinyimages_prediction.py --model_path ../cifar10-vs-ti/model_state_epoch520.pth --data_dir ../data --output_dir ../data/ --output_filename ti_500K_unlabeled.pickle  
 ```  
 
### Step four: Training a vanilla model on CIFAR-10 
We now train a model (Wide ResNet 28-10) on CIFAR-10 training set.   
  
 ```
python robust_self_training.py --distance l_2 --beta 0 --unsup_fraction 0 --model_dir vanilla  
 ```

### Step five: Generating pseudo-labels 
As a final step, we generate pseudo-labels by applying the classifier from Step 4 on the unlabeled data sourced in Step 3.   
  
 ```
python generate_pseudolabels.py --model_dir ../vanilla  --model_epoch 200 --data_dir ../data/ --data_filename ti_500K_unlabeled.pickle --output_dir ../data/ --output_filename ti_500K_pseudo_labeled.pickle  
 ``` 

 ## Reference  
```  
@inproceedings{carmon2019unlabeled,  
author = {Yair Carmon and Aditi Raghunathan and Ludwig Schmidt and Percy Liang and John Duchi},  
title = {Unlabeled Data Improves Adversarial Robustness},  
year = 2019,  
booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},  
}  
```