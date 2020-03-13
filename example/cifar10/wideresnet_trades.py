''' This file provides a wrapper class for TRADES (https://github.com/yaodongyu/TRADES) model for CIFAR-10 dataset. '''

import os
import torch
import tensorflow as tf

from realsafe.model.pytorch_wrapper import pytorch_classifier_with_logits
from realsafe.model.loader import get_res_path

from third_party.wideresnet import WideResNet

MODEL_PATH = get_res_path('./cifar10/wrn.pt')


def load(_):
    model = WideResNet_TRADES()
    model.load()
    return model


@pytorch_classifier_with_logits(10, 0.0, 1.0, (32, 32, 3), tf.float32, tf.int32)
class WideResNet_TRADES(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.model = WideResNet().cuda()

    def forward(self, x):
        x = x.transpose(1, 2).transpose(1, 3).contiguous()
        labels = self.model(x.cuda())
        return labels.cpu()

    def load(self):
        checkpoint = torch.load(MODEL_PATH)
        self.model.load_state_dict(checkpoint)
        self.eval()


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        if not os.path.exists(os.path.dirname(MODEL_PATH)):
            os.makedirs(os.path.dirname(MODEL_PATH))
        url = 'https://drive.google.com/file/d/10sHvaXhTNZGz618QmD5gSOAjO3rMzV33/view'
        print('Please download "{}" to "{}".'.format(url, MODEL_PATH))
