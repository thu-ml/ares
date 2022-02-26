import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pytorch_ares.attack_torch.utils import loss_adv


class SI_NI_FGSM(object):
    '''Projected Gradient Descent'''
    def __init__(self, net, epsilon, p, scale_factor, stepsize, decay_factor, steps, data_name,target, loss, device):
        self.epsilon = epsilon
        self.p = p
        self.net = net
        self.scale_factor = scale_factor
        self.decay_factor = decay_factor
        self.stepsize = stepsize
        self.target = target
        self.steps = steps
        self.loss = loss
        self.data_name = data_name
        self.device = device
        if self.data_name=="cifar10" and self.target:
            raise AssertionError('cifar10 dont support targeted attack')

    
    def forward(self, image, label, target_labels):
        image, label = image.to(self.device), label.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        batchsize = image.shape[0]
        advimage = image
        # PGD to get adversarial example
        momentum = torch.zeros_like(image).detach()
        for i in range(self.steps):
            advimage_nes = advimage + self.decay_factor * self.stepsize * momentum
            grads = torch.zeros_like(image).to(self.device)
            for j in range(self.scale_factor):
                x_s = (advimage_nes / 2**(j)).requires_grad_(True)
                netOut = self.net(x_s)
                loss = loss_adv(self.loss, netOut, label, target_labels, self.target, self.device)     
                loss.backward(retain_graph=True)
                grads += torch.autograd.grad(loss, [x_s])[0].detach()
             # clone the advimage as the next iteration input

            grads_norm = torch.norm(nn.Flatten()(grads), p=1, dim=1) 
            grads = grads / grads_norm.view([-1]+[1]*(len(grads.shape)-1))
            grads = self.decay_factor * momentum + grads
            momentum = grads
    
            if self.p==np.inf:
                updates = grads.sign()
            else:
                normVal = torch.norm(grads.view(batchsize, -1), self.p, 1)
                updates = grads/normVal.view(batchsize, 1, 1, 1)
            updates = updates*self.stepsize
            advimage = advimage + updates
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
            
            advimage= torch.clamp(advimage, 0, 1)#cifar10(-1,1)
           
        return advimage