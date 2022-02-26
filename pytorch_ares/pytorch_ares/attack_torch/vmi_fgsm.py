import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pytorch_ares.attack_torch.utils import loss_adv


class VMI_fgsm(object):
    '''Projected Gradient Descent'''
    def __init__(self, net, epsilon, p, beta, sample_number, stepsize, steps, decay_factor, data_name,target, loss, device):
        self.epsilon = epsilon
        self.p = p
        self.beta = beta
        self.sample_number = sample_number
        self.net = net
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
        momentum = torch.zeros_like(image).detach()
        variance = torch.zeros_like(image).detach()
        # PGD to get adversarial example

        for i in range(self.steps):
            advimage = advimage.clone().detach().requires_grad_(True) # clone the advimage as the next iteration input
            
            
            netOut = self.net(advimage)
            loss = loss_adv(self.loss, netOut, label, target_labels, self.target, self.device)      
            gradpast = torch.autograd.grad(loss, [advimage])[0].detach()
            grad = momentum * self.decay_factor + (gradpast + variance) / torch.norm(gradpast + variance, p=1)

            #update variance
            sample = advimage.clone().detach()
            global_grad = torch.zeros_like(image).detach()
            for j in range(self.sample_number):
                sample = sample.detach()
                sample.requires_grad = True
                randn = (torch.rand_like(image) * 2 - 1) * self.beta * self.epsilon
                sample = sample + randn
                sample_norm = (sample - self.mean_torch) / self.std_torch
                outputs_sample = self.net(sample_norm)
                loss = loss_adv(self.loss, outputs_sample, label, target_labels, self.target, self.device) 
                global_grad += torch.autograd.grad(loss, sample, grad_outputs=None, only_inputs=True)[0]
            variance = global_grad / (self.sample_number * 1.0) - gradpast
  
            momentum = grad
            if self.p==np.inf:
                updates = grad.sign()
            else:
                normVal = torch.norm(grad.view(batchsize, -1), self.p, 1)
                updates = grad/normVal.view(batchsize, 1, 1, 1)
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