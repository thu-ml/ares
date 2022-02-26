"""adversary.py"""
import numpy as np
from scipy import stats as st
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_ares.attack_torch.utils import loss_adv


class TIFGSM(object):
    def __init__(self, net, p, kernel_name, len_kernel, nsig, eps, stepsize, steps, decay, resize_rate, diversity_prob, data_name,target,loss, device):
        self.net = net
        self.epsilon = eps
        self.p = p
        self.steps = steps
        self.decay = decay
        self.target = target
        self.loss = loss
        self.stepsize = stepsize
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.data_name = data_name
        self.device = device
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        if self.data_name=="cifar10" and self.target:
            raise AssertionError('cifar10 dont support targeted attack')

    
    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)
        
        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]
            
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x


    def forward(self, image, labels, target_labels):

        images = image.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if target_labels is not None:
            target_labels = target_labels.clone().detach().to(self.device)
        batchsize = image.shape[0]
        momentum = torch.zeros_like(images).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

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
            advimage = advimage.clone().detach().requires_grad_(True)
            
            outputs = self.net(self.input_diversity(advimage))

            loss = loss_adv(self.loss, outputs, labels, target_labels, self.target, self.device)   
            grad = torch.autograd.grad(loss, [advimage])[0].detach()
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding=int((self.len_kernel-1)/2), groups=3)
            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
            grad = grad + momentum*self.decay
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

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel