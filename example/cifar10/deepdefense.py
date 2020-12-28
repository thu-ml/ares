'''
This file provides a wrapper class for deep defense (https://github.com/ZiangYan/deepdefense.pytorch) model for CIFAR-10
dataset.
'''

import sys
import os

THIRD_PARTY_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party'))
if THIRD_PARTY_PATH not in sys.path:
    sys.path.append(THIRD_PARTY_PATH)

from deepdefense_pytorch.models.cifar10 import ConvNet

from ares.model import ClassifierWithLogits
from ares.utils import get_res_path, download_res
from ares.model.pytorch_wrapper import pytorch_classifier_with_logits

import torch
import scipy.io
import numpy as np
import tensorflow as tf

MODEL_PATH = get_res_path('./cifar10/deepdefense')


def load(_):
    model = ConvNet_DeepDefense()
    model.load(MODEL_PATH)
    return model


def download(model_path):
    mat_name = 'cifar10-convnet-15742544.mat'
    mat_path = os.path.abspath(os.path.join(model_path, mat_name))
    tar_path = os.path.abspath(os.path.join(model_path, 'mnist-cifar10-data-model.tar'))
    if not os.path.exists(mat_path):
        if not os.path.exists(os.path.dirname(tar_path)):
            os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        if not os.path.exists(tar_path):
            print('Please download "mnist-cifar10-data-model.tar" from ' +
                  '"https://drive.google.com/open?id=15xoZ-LUbc9GZpTlxmCJmvL_DR2qYEu2J", ' +
                  'and save it to "{}".'.format(tar_path))
            return
        import tarfile
        with tarfile.TarFile(tar_path) as f:
            mat_file = f.extractfile('data/' + mat_name)
            with open(mat_path, 'wb+') as t:
                t.write(mat_file.read())
        os.remove(tar_path)
    stats_path = os.path.abspath(os.path.join(model_path, 'stats.mat'))
    if not os.path.exists(stats_path):
        download_res('http://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/ares/DeepDefense/stats.mat', stats_path)


@pytorch_classifier_with_logits(10, 0.0, 255.0, (32, 32, 3), tf.float32, tf.int32)
class ConvNet_DeepDefense(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self._model = ConvNet().cuda()

    def forward(self, x):
        x = x - self._mean
        x = x.transpose(1, 3).reshape((x.shape[0], -1))
        x = x - x.mean(dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        x = x * 53.0992088 / std.clamp(min=40).reshape((-1, 1))
        input_var = torch.mm(x, self._trans)
        input_var = input_var.reshape((input_var.shape[0], 3, 32, 32))
        labels = self._model(input_var.cuda())
        return labels.cpu()

    def load_stats(self, model_path):
        stats = scipy.io.loadmat(os.path.join(model_path, 'stats.mat'))
        self._mean = torch.from_numpy(stats['dataMean'][np.newaxis])
        self._trans = torch.from_numpy(stats['Trans'].T)

    def load_weights(self, model_path):
        mcn = scipy.io.loadmat(os.path.join(model_path, 'cifar10-convnet-15742544.mat'))
        mcn_weights = dict()
        mcn_weights['conv1.weights'] = mcn['net'][0][0][0][0][0][0][0][1][0][0].transpose()
        mcn_weights['conv1.bias'] = mcn['net'][0][0][0][0][0][0][0][1][0][1].flatten()
        mcn_weights['conv2.weights'] = mcn['net'][0][0][0][0][3][0][0][1][0][0].transpose()
        mcn_weights['conv2.bias'] = mcn['net'][0][0][0][0][3][0][0][1][0][1].flatten()
        mcn_weights['conv3.weights'] = mcn['net'][0][0][0][0][6][0][0][1][0][0].transpose()
        mcn_weights['conv3.bias'] = mcn['net'][0][0][0][0][6][0][0][1][0][1].flatten()
        mcn_weights['conv4.weights'] = mcn['net'][0][0][0][0][9][0][0][1][0][0].transpose()
        mcn_weights['conv4.bias'] = mcn['net'][0][0][0][0][9][0][0][1][0][1].flatten()
        mcn_weights['conv5.weights'] = mcn['net'][0][0][0][0][11][0][0][1][0][0].transpose()
        mcn_weights['conv5.bias'] = mcn['net'][0][0][0][0][11][0][0][1][0][1].flatten()

        for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            t = self._model.__getattr__(k)
            assert t.weight.data.size() == mcn_weights['%s.weights' % k].shape
            t.weight.data[:] = torch.from_numpy(mcn_weights['%s.weights' % k])
            assert t.bias.data.size() == mcn_weights['%s.bias' % k].shape
            t.bias.data[:] = torch.from_numpy(mcn_weights['%s.bias' % k])

    def load(self, model_path):
        self.load_stats(model_path)
        self.load_weights(model_path)


if __name__ == '__main__':
    download(MODEL_PATH)
