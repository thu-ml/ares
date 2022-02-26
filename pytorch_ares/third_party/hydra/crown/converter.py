## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
##
import sys
import copy
import torch
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
import numpy as np
from datasets import loaders
from model_defs import Flatten, model_mlp_any, model_cnn_1layer, model_cnn_2layer, model_cnn_4layer, model_cnn_3layer
from bound_layers import BoundSequential
import torch.optim as optim
import time
from datetime import datetime

from config import load_config, get_path, config_modelloader, config_dataloader, config_modelloader_and_convert2mlp
from argparser import argparser
from pdb import set_trace as st
# sys.settrace(gpu_profile)


def main(args):
    config = load_config(args)
    global_train_config = config["training_params"]
    models, model_names = config_modelloader_and_convert2mlp(config)

if __name__ == "__main__":
    args = argparser()
    main(args)
