"""adversary.py"""
# from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
# from attack_torch.utils import Normalize

class PGD(object):
    '''Projected Gradient Descent'''
    def __init__(self, net, epsilon, norm, stepsize, steps,target, device):
        self.epsilon = epsilon
        self.p = norm
        self.net = net
        self.stepsize = stepsize
        self.steps = steps
        self.target = target
        self.device = device
        if self.target:
            raise AssertionError('cifar10 dont support targeted attack')

    
    def ce_loss(self, outputs, labels):
        loss = nn.CrossEntropyLoss()
        
        if self.target:
            raise AssertionError('cifar10 dont support targeted attack')
        else:
            cost = loss(outputs, labels)
        return cost
    
    def forward(self, image, label):
        image, label = image.to(self.device), label.to(self.device)
        batchsize = image.shape[0]
        # random start
        delta = torch.rand_like(image)*2*self.epsilon-self.epsilon
        if self.p!=np.inf: # projected into feasible set if needed
            normVal = torch.norm(delta.view(batchsize, -1), self.p, 1)#求范数
            mask = normVal<=self.epsilon
            scaling = self.epsilon/normVal
            scaling[mask] = 1
            delta = delta*scaling.view(batchsize, 1, 1, 1)
        advimage = image+delta
        

        for i in range(self.steps):
            advimage = advimage.clone().detach().requires_grad_(True) # clone the advimage as the next iteration input
            netOut = self.net(advimage)
            
            loss = self.ce_loss(netOut, label)
                  
            updates = torch.autograd.grad(loss, [advimage])[0].detach()
            if self.p==np.inf:
                updates = updates.sign()
            else:
                normVal = torch.norm(updates.view(batchsize, -1), self.p, 1)
                updates = updates/normVal.view(batchsize, 1, 1, 1)
            updates = updates*self.stepsize
            advimage = advimage+updates
            # project the disturbed image to feasible set if needed
            delta = advimage-image
            if self.p==np.inf:
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            else:
                normVal = torch.norm(delta.view(batchsize, -1), self.p, 1)
                mask = normVal<=self.epsilon
                scaling = self.epsilon/normVal
                scaling[mask] = 1
                delta = delta*scaling.view(batchsize, 1, 1, 1)
            advimage = image+delta
            
            advimage = torch.clamp(advimage, 0, 1)#cifar10(-1,1)
            
        return advimage