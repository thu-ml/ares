''' This file provides a wrapper class for Hydra (https://github.com/inspire-group/hydra) model for CIFAR-10 dataset. '''

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import math

from ares.model.pytorch_wrapper import pytorch_classifier_with_logits
from ares.utils import get_res_path

MODEL_PATH = get_res_path('./cifar10/model_best_dense.pth.tar')


def load(_):
    model = Hydra()
    model.load()
    return model


@pytorch_classifier_with_logits(n_class=10, x_min=0.0, x_max=1.0,
                                x_shape=(32, 32, 3), x_dtype=tf.float32, y_dtype=tf.int32)
class Hydra(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.model = nn.DataParallel(wrn_28_10()).cuda()

    def forward(self, x):
        x = x.transpose(1, 2).transpose(1, 3).contiguous()
        labels = self.model(x.cuda())
        return labels.cpu()

    def load(self):
        checkpoint = torch.load(MODEL_PATH, map_location="cuda")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()


class BasicBlock(nn.Module):
    def __init__(self, conv_layer, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv_layer(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and conv_layer(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(
        self, nb_layers, in_planes, out_planes, block, conv_layer, stride, dropRate=0.0
    ):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            conv_layer, block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(
        self, conv_layer, block, in_planes, out_planes, nb_layers, stride, dropRate
    ):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    conv_layer,
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        conv_layer,
        linear_layer,
        depth=34,
        num_classes=10,
        widen_factor=10,
        dropRate=0.0,
    ):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = conv_layer(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, conv_layer, 1, dropRate
        )
        # 1st sub-block
        self.sub_block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, conv_layer, 1, dropRate
        )
        # 2nd block
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], block, conv_layer, 2, dropRate
        )
        # 3rd block
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], block, conv_layer, 2, dropRate
        )
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = linear_layer(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, linear_layer):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def wrn_28_10(**kwargs):
    return WideResNet(nn.Conv2d, nn.Linear, depth=28, widen_factor=10, **kwargs)


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        if not os.path.exists(os.path.dirname(MODEL_PATH)):
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = 'https://www.dropbox.com/sh/56yyfy16elwbnr8/AADmr7bXgFkrNdoHjKWwIFKqa?dl=0&preview=model_best_dense.pth.tar'
        print('Please download "{}" to "{}".'.format(url, MODEL_PATH))
