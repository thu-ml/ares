import torch
import numpy as np
import torch.nn as nn
from ares.attack.pgd import PGD
from ares.attack.mim import MIM
from ares.utils.registry import registry

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


@registry.register_attack('sgm')
class SGM(object):
    '''Skip Gradient Method. A transfer-based black-box attack method.

    Example:
        >>> from ares.utils.registry import registry
        >>> attacker_cls = registry.get_attack('sgm')
        >>> attacker = attacker_cls(model)
        >>> adv_images = attacker(images, labels, target_labels)

    - Supported distance metric: 1, 2, np.inf.
    - References: https://arxiv.org/abs/2002.05990.
    '''
    def __init__(self, model, device='cuda', norm=np.inf, eps=4/255, stepsize=1/255, steps=20, net_name='resnet50',
                 gamma=0.0, momentum=1.0, loss='ce', target=False):
        '''The initialize function for SGM.

        Args:
            model (torch.nn.Module): The target model to be attacked.
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            norm (float): The norm of distance calculation for adversarial constraint. Defaults to np.inf.
            eps (float): The maximum perturbation range epsilon.
            stepsize (float): The attack range for each step.
            steps (float): The number of attack iteration.
            net_name (str): The name of the network architecture.
            gamma (float): The parameter gamma.
            momentum (float): The momentum for attack optimizer.
            loss (str): The loss function.
            target (bool): Conduct target/untarget attack. Defaults to False.
        '''

        self.epsilon = eps
        self.p = norm
        self.net = model
        self.net_name=net_name
        self.stepsize = stepsize
        self.steps = steps
        self.gamma=gamma
        self.momentum=momentum
        self.loss = loss
        self.target = target
        self.device = device
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
    
    def __call__(self, images=None, labels=None, target_labels=None):
        '''This function perform attack on target images with corresponding labels 
        and target labels for target attack.

        Args:
            images (torch.Tensor): The images to be attacked. The images should be torch.Tensor with shape [N, C, H, W] and range [0, 1].
            labels (torch.Tensor): The corresponding labels of the images. The labels should be torch.Tensor with shape [N, ]
            target_labels (torch.Tensor): The target labels for target attack. The labels should be torch.Tensor with shape [N, ]

        Returns:
            torch.Tensor: Adversarial images with value range [0,1].

        '''
        images, labels = images.to(self.device), labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        
        advimage=self.adversary.forward(images, labels, target_labels)
        
        return advimage
