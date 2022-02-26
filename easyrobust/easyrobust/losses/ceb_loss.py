'''

CEB Improves Model Robustness
Paper link: https://arxiv.org/abs/2002.05380

Modified by Xiaofeng Mao 
2021.8.30
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical

def lerp(global_step, start_step, end_step, start_val, end_val):
    """Utility function to linearly interpolate two values."""
    interp = (global_step - start_step) / (end_step - start_step)
    interp = max(0.0, min(1.0, interp))
    return start_val * (1.0 - interp) + end_val * interp

class Conditional_Entropy_Bottleneck(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, num_classes, end_rho=3.0, start_rho=100.0, anneal_rho=120000):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(Conditional_Entropy_Bottleneck, self).__init__()

        self.y_project = nn.Linear(num_classes, 2048)
        self.cls_project = nn.Linear(2048, num_classes)
        self.num_classes = num_classes
        self.rho_to_gamma = lambda rho: 1.0 / np.exp(rho)
        self.end_rho = end_rho
        self.start_rho = start_rho
        self.anneal_rho = anneal_rho
        self.global_step = 0

    def forward(self, feature, target):
        # feature: b*2048
        m = MultivariateNormal(feature, torch.eye(feature.size(1)).cuda())
        if self.training:
            z = m.sample()
        else:
            z = m.mean()

        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=self.num_classes)
        y_emb = self.y_project(one_hot_target.float())

        y_emb_d = MultivariateNormal(y_emb, torch.eye(y_emb.size(1)).cuda())
        logits_d = Categorical(self.cls_project(z))
        logits = logits_d.logits

        h1 = -m.log_prob(z)
        h2 = -y_emb_d.log_prob(z)
        h3 = -logits_d.log_prob(target)

        rex = -h1 + h2

        if self.anneal_rho > 0:
            gamma = lerp(self.global_step, 0, self.anneal_rho, self.rho_to_gamma(self.start_rho), self.rho_to_gamma(self.end_rho))
            loss = torch.mean(gamma * rex + h3)
        self.global_step += 1
        
        return logits, loss