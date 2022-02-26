"""adversary.py"""
# from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
from pytorch_ares.attack_torch import *
# from attack_torch.utils import Normalize

def backward_hook(gamma):
    # implement SGM through grad through ReLU
    def _backward_hook(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_in[0],)
    return _backward_hook


def backward_hook_norm(module, grad_in, grad_out):
    # normalize the gradient to avoid gradient explosion or vanish
    std = torch.std(grad_in[0])
    return (grad_in[0] / std,)


def register_hook_for_resnet(model, arch, gamma):
    # There is only 1 ReLU in Conv module of ResNet-18/34
    # and 2 ReLU in Conv module ResNet-50/101/152
    if arch in ['resnet50', 'resnet101', 'resnet152']:
        gamma = np.power(gamma, 0.5)
    print('gamma:')
    print(gamma)
    backward_hook_sgm = backward_hook(gamma)

    for name, module in model.named_modules():
        # print('name:')
        # print(name)
        # print(isinstance(module, nn.ReLU))

        if 'act' in name and not '0.act' in name:
            module.register_backward_hook(backward_hook_sgm)

        # e.g., 1.layer1.1, 1.layer4.2, ...
        # if len(name.split('.')) == 3:
        if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
            module.register_backward_hook(backward_hook_norm)


def register_hook_for_densenet(model, arch, gamma):
    # There are 2 ReLU in Conv module of DenseNet-121/169/201.
    gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)
    for name, module in model.named_modules():
        if 'relu' in name and not 'transition' in name:
            module.register_backward_hook(backward_hook_sgm)



class SGM(object):
    '''Skip Gradient Method'''
    def __init__(self, net, net_name, epsilon, norm, stepsize, steps, gamma, momentum, data_name,target,loss, device):
        self.epsilon = epsilon
        self.p = norm
        self.net = net
        self.net_name=net_name
        self.stepsize = stepsize
        self.steps = steps
        self.gamma=gamma
        self.momentum=momentum
        self.loss = loss
        self.target = target
        self.data_name = data_name
        self.device = device
        if self.data_name=="cifar10" and self.target:
            raise AssertionError('cifar10 dont support targeted attack')
        if self.gamma < 1.0:
            if self.net_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
                register_hook_for_resnet(self.net, arch=self.net_name, gamma=self.gamma)
            elif self.net_name in ['densenet121', 'densenet169', 'densenet201']:
                register_hook_for_densenet(self.net, arch=self.net_name, gamma=self.gamma)
            else:
                raise ValueError('Current code only supports resnet/densenet. '
                                'You can extend this code to other architectures.')
        
        if self.momentum > 0.0:
            print('using PGD attack with momentum = {}'.format(self.momentum))
            self.adversary = MIM(net=self.net, epsilon=self.epsilon, p=self.p, stepsize=self.stepsize, steps=self.steps, decay_factor=self.momentum,
                data_name=self.data_name,target=self.target,loss=self.loss, device=self.device)
        else:
            if self.p==np.inf or 2:
                print('using PGD attack')
                self.adversary = PGD(net=self.net, epsilon=self.epsilon, norm=self.p, stepsize=self.stepsize, steps=self.steps,
                    data_name=self.data_name,target=self.target,loss=self.loss, device=self.device)
            else:
                raise ValueError('SGM uses PGD attacker.')
    
    def forward(self, image, label, target_labels):
        image, label = image.to(self.device), label.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        
        advimage=self.adversary.forward(image, label, target_labels)
        
        return advimage