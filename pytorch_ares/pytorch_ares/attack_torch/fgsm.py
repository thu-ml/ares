"""adversary.py"""
import torch
import numpy as np
import torch.nn as nn
from pytorch_ares.attack_torch.utils import loss_adv

class FGSM(object):
    def __init__(self, net, p, eps, data_name,target,loss, device):
        self.net = net
        self.eps = eps
        self.p = p
        self.target = target
        self.data_name = data_name
        self.loss = loss
        self.device = device
        if self.data_name=="cifar10" and self.target:
            raise AssertionError('cifar10 dont support targeted attack')

    
    def forward(self, images, labels,target_labels):
        batchsize = images.shape[0]
        images, labels = images.to(self.device), labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        advimage = images.clone().detach().requires_grad_(True).to(self.device)
        outputs = self.net(advimage)
            
    
        loss = loss_adv(self.loss, outputs, labels, target_labels, self.target, self.device) 
             
        updatas = torch.autograd.grad(loss, [advimage])[0].detach()

        if self.p == np.inf:
            updatas = updatas.sign()
        else:
            normval = torch.norm(updatas.view(batchsize, -1), self.p, 1)
            updatas = updatas / normval.view(batchsize, 1, 1, 1)
        
        advimage = advimage + updatas*self.eps
        delta = advimage - images

        if self.p==np.inf:
            delta = torch.clamp(delta, -self.eps, self.eps)
        else:
            normVal = torch.norm(delta.view(batchsize, -1), self.p, 1)
            mask = normVal<=self.eps
            scaling = self.eps/normVal
            scaling[mask] = 1
            delta = delta*scaling.view(batchsize, 1, 1, 1)
        advimage = images+delta
        
        advimage = torch.clamp(advimage, 0, 1)
        
        return advimage

