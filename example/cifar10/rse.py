'''
This file provides a wrapper class for random self-ensemble (https://github.com/xuanqing94/BayesianDefense) model for
CIFAR-10 dataset.
'''

import sys
import os

THIRD_PARTY_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party'))
if THIRD_PARTY_PATH not in sys.path:
    sys.path.append(THIRD_PARTY_PATH)

from bayesian_defense.models.vgg_rse import VGG

import torch
import tensorflow as tf

from realsafe.utils import get_res_path, download_res
from realsafe.model.pytorch_wrapper import pytorch_classifier_with_logits

MODEL_PATH = get_res_path('./cifar10/rse')


def load(_):
    model = VGG_RSE()
    model.load(MODEL_PATH)
    return model


def download(model_path):
    pth_path = os.path.join(model_path, 'cifar10_vgg_rse.pth')
    if not os.path.exists(os.path.dirname(pth_path)):
        os.makedirs(os.path.dirname(pth_path))
    if not os.path.exists(pth_path):
        print('Please download "cifar10_vgg_rse.pth" from ' +
              '"https://drive.google.com/drive/folders/1uAP6q-fSADhlkx-YNBipPzv4KCvC_3_J?usp=sharing", ' +
              'and save it to "{}".'.format(pth_path))


@pytorch_classifier_with_logits(10, 0.0, 1.0, (32, 32, 3), tf.float32, tf.int32)
class VGG_RSE(torch.nn.Module):
    def __init__(self, use_cuda=True):
        torch.nn.Module.__init__(self)
        self._model = VGG('VGG16', nclass=10, noise_init=0.2, noise_inner=0.1, img_width=32).cuda()
        self._num_ensemble = 50

    def forward(self, x):
        x = x.transpose(1, 2).transpose(1, 3).contiguous()
        x_ex = x.unsqueeze(1).repeat(1, self._num_ensemble, 1, 1, 1) .view(-1, x.shape[1], x.shape[2], x.shape[3])
        labels = self._model(x_ex.cuda())[0]
        labels = labels.view(x.shape[0], self._num_ensemble, -1).mean(dim=1)
        return labels.cpu()

    def load(self, model_path):
        pth_path = os.path.join(model_path, 'cifar10_vgg_rse.pth')
        checkpoint = torch.load(pth_path)
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]
            state_dict[name] = v
        self._model.load_state_dict(state_dict)
        self.eval()


if __name__ == '__main__':
    download(MODEL_PATH)
