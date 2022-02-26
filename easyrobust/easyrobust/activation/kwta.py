'''

Resisting Adversarial Attacks by k-Winners-Take-All (ICLR2020)
Paper link: https://arxiv.org/abs/1905.10510

Modified by Xiaofeng Mao 
2021.7.22
'''

from torch import nn as nn

class kWTA(nn.Module):
    def __init__(self):
        super(kWTA, self).__init__()

    def forward(self, x):
        size = x.shape[1]*x.shape[2]*x.shape[3]
        k = int(self.sr*size)
        tmpx = x.view(x.shape[0], -1)
        topval = tmpx.topk(k, dim=1)[0][:,-1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1,0).view_as(x)
        comp = (x>=topval).to(x)
        return comp*x
