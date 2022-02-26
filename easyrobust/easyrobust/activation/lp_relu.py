'''

Robust Image Classification Using a Low-Pass Activation Function and DCT Augmentation
Paper link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9455411

Modified by Xiaofeng Mao 
2021.8.4
'''

from torch import nn as nn
import torch

class LP_ReLU1(nn.Module):
    def __init__(self, inplace=False):
        super(LP_ReLU1, self).__init__()
        self.thre = torch.tensor(0).cuda()
        self.thre2 = torch.tensor(10).cuda()

    def forward(self, x):
        return torch.minimum(torch.maximum(self.thre, x), self.thre2) + torch.maximum(self.thre, 0.05*(x-10))

class LP_ReLU2(nn.Module):
    def __init__(self, inplace=False):
        super(LP_ReLU2, self).__init__()
        self.thre = torch.tensor(0).cuda()
        self.thre2 = torch.tensor(5).cuda()
        self.thre3 = torch.tensor(0.15).cuda()

    def forward(self, x):
        return torch.minimum(torch.maximum(self.thre, x), self.thre2) + torch.minimum(torch.maximum(self.thre, 0.05*(x-5)), self.thre3) + torch.maximum(self.thre, 0.02*(x-8))

