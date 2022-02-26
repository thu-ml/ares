'''

Gaussian-Based Pooling for Convolutional Neural Networks (NeurIPS2019)
Paper link: https://staff.aist.go.jp/takumi.kobayashi/publication/2019/NeurIPS2019.pdf

Modified by Xiaofeng Mao 
2021.9.16
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianPooling2d(nn.AvgPool2d):
    def __init__(self, num_features, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, hidden_node=None, stochasticity='HWCN', eps=1e-6):
        if stochasticity != 'HWCN' and stochasticity != 'CN' and stochasticity is not None:
            raise ValueError("gaussian pooling stochasticity has to be 'HWCN'/'CN' or None, "
                         "but got {}".format(stochasticity))
        if hidden_node is None:
            hidden_node = num_features // 2

        super(GaussianPooling2d, self).__init__(kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                    count_include_pad=count_include_pad)
        self.eps = eps
        self.stochasticity = stochasticity

        self.ToHidden = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_features, hidden_node, kernel_size=1,  padding=0, bias=True),
            nn.BatchNorm2d(hidden_node),
            nn.ReLU(False),
        )
        self.ToMean = nn.Sequential(
            nn.Conv2d(hidden_node, num_features, kernel_size=1,  padding=0, bias=True),
            nn.BatchNorm2d(num_features),
        )
        self.ToSigma = nn.Sequential(
            nn.Conv2d(hidden_node, num_features, kernel_size=1,  padding=0, bias=True),
            nn.BatchNorm2d(num_features),
            nn.Sigmoid()
        )
        self.activation = nn.Softplus()
        
    def forward(self, input):
        mu0 = F.avg_pool2d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
        sig0= F.avg_pool2d(input**2, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
        sig0= torch.sqrt(torch.clamp(sig0 - mu0**2, self.eps))

        Z = self.ToHidden(input)
        MU = self.ToMean(Z)

        if self.training and self.stochasticity is not None:
            SIGMA = self.ToSigma(Z)
            if self.stochasticity == 'HWCN':
                size = sig0.size()
            else:
                size = [sig0.size(0), sig0.size(1), 1, 1]
            W = self.activation(MU + SIGMA * 
                torch.randn(size, dtype=sig0.dtype, layout=sig0.layout, device=sig0.device))
        else:
            W = self.activation(MU)

        return mu0 + W*sig0