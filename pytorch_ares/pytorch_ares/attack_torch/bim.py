"""adversary.py"""
# from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
from pytorch_ares.attack_torch.utils import loss_adv

class BIM(object):
    '''Projected Gradient Descent'''
    def __init__(self, net, epsilon, p, stepsize, steps, data_name,target,loss, device):
        self.epsilon = epsilon
        self.p = p
        self.net = net
        self.stepsize = stepsize
        self.steps = steps
        self.target = target
        self.data_name = data_name
        self.loss = loss
        self.device = device
        # assert self.data_name=="cifar10" and self.target
        if self.data_name=="cifar10" and self.target:
            raise AssertionError('cifar10 dont support targeted attack')

    
    def forward(self, images, labels,target_labels):
        batchsize = images.shape[0]
        images, labels = images.to(self.device), labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        advimage = images
       
        

        for i in range(self.steps):
            advimage = advimage.clone().detach().requires_grad_(True).to(self.device)
            netOut = self.net(advimage)
            loss = loss_adv(self.loss, netOut, labels, target_labels, self.target, self.device) 
            updates = torch.autograd.grad(loss, [advimage])[0].detach()
            if self.p==np.inf:
                updates = updates.sign()
            else:
                normVal = torch.norm(updates.view(batchsize, -1), self.p, 1)
                updates = updates/normVal.view(batchsize, 1, 1, 1)
            updates = updates*self.stepsize
            
            advimage = advimage + updates
            
            delta = advimage-images
            if self.p==np.inf:
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            else:
                normVal = torch.norm(delta.view(batchsize, -1), self.p, 1)
                mask = normVal<=self.epsilon
                scaling = self.epsilon/normVal
                scaling[mask] = 1
                delta = delta*scaling.view(batchsize, 1, 1, 1)
            advimage = images+delta
            
            advimage = torch.clamp(advimage, 0, 1)
          
        return advimage