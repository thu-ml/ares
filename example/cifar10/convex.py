'''
This file provides a wrapper class for convex (https://github.com/locuslab/convex_adversarial/) model for CIFAR-10
dataset.
'''

import sys
import os

THIRD_PARTY_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party'))
if THIRD_PARTY_PATH not in sys.path:
    sys.path.append(THIRD_PARTY_PATH)

MODULE_PATH = '../../third_party/convex_adversarial_pytorch'
MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), MODULE_PATH))
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from convex_adversarial_pytorch.examples.problems import cifar_model_resnet, Flatten
from convex_adversarial.utils import Dense, DenseSequential

sys.path.remove(MODULE_PATH)

import torch
import numpy as np
import tensorflow as tf

from ares.utils import get_res_path, download_res
from ares.model.pytorch_wrapper import pytorch_classifier_with_logits

MODEL_PATH = get_res_path('./cifar10/convex')


def load(_):
    model = ResNet_Convex()
    model.load(MODEL_PATH)
    return model


def download(model_path):
    pth_path = os.path.join(model_path, 'cifar_resnet_2px.pth')
    if not os.path.exists(pth_path):
        if not os.path.exists(os.path.dirname(pth_path)):
            os.makedirs(os.path.dirname(pth_path), exist_ok=True)
        url = 'https://github.com/locuslab/convex_adversarial/blob/master/models_scaled/cifar_resnet_2px.pth?raw=true'
        download_res(url, pth_path)


@pytorch_classifier_with_logits(10, 0.0, 1.0, (32, 32, 3), tf.float32, tf.int32)
class ResNet_Convex(torch.nn.Module):
    def __init__(self, use_cuda=True):
        torch.nn.Module.__init__(self)
        self._model = cifar_model_resnet(N=1, factor=1).cuda()
        self._mean_torch = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).astype(np.float32))
        self._std_torch = torch.from_numpy(np.array([0.225, 0.225, 0.225]).reshape([1, 3, 1, 1]).astype(np.float32))

    def forward(self, x):
        x = x.transpose(1, 2).transpose(1, 3).contiguous()
        input_var = (x - self._mean_torch) / self._std_torch
        labels = self._model(input_var.cuda())
        return labels.cpu()

    def load(self, model_path):
        pth_path = os.path.join(model_path, 'cifar_resnet_2px.pth')
        checkpoint = torch.load(pth_path)
        self._model.load_state_dict(checkpoint['state_dict'][0])


if __name__ == '__main__':
    download(MODEL_PATH)
