"""

evaluate simple transferabl attacks in the single-model transfer setting.

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
# from tqdm import tqdm, tqdm_notebook
import numpy as np
import os
import scipy.stats as st


##define TI
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel
def TI_tta(kernel_size=5, nsig=3):
    channels=3
    kernel = gkern(kernel_size, nsig).astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
    gaussian_kernel = torch.from_numpy(gaussian_kernel)
    return gaussian_kernel


## define Po+Trip
def Poincare_dis(a, b):
    L2_a = torch.sum(torch.square(a), 1)
    L2_b = torch.sum(torch.square(b), 1)

    theta = 2 * torch.sum(torch.square(a - b), 1) / ((1 - L2_a) * (1 - L2_b))
    distance = torch.mean(torch.acosh(1.0 + theta))
    return distance

def Cos_dis(a, b):
    a_b = torch.abs(torch.sum(torch.multiply(a, b), 1))
    L2_a = torch.sum(torch.square(a), 1)
    L2_b = torch.sum(torch.square(b), 1)
    distance = torch.mean(a_b / torch.sqrt(L2_a * L2_b))
    return distance

class TTA(object):
    '''targeted transfer attack'''
    def __init__(self, net, epsilon, norm, stepsize, steps, kernel_size, nsig, resize_rate, diversity_prob, data_name,target, loss, device):
        self.epsilon = epsilon
        self.p = norm
        self.net = net
        self.stepsize = stepsize
        self.steps = steps
        self.loss = loss
        self.target = target
        self.data_name = data_name
        self.device = device
        self.resize_rate=resize_rate
        self.diversity_prob=diversity_prob
        self.gaussian_kernel=TI_tta(kernel_size, nsig).to(device)
    
    def ce_loss(self, outputs, labels, target_labels):
        loss = nn.CrossEntropyLoss()
        
        if self.target:
            cost = loss(outputs, target_labels)
        else:
            cost = -loss(outputs, labels)
        return cost


    def logits_loss(self, outputs, labels, target_labels):
        loss=None
        if self.target:
            real = outputs.gather(1,target_labels.unsqueeze(1)).squeeze(1)
            logit_dists = ( -1 * real)
            loss = logit_dists.sum()
        else:
            raise Exception('Untarget attack not supported in logits loss.')
        return loss
    
    def po_trip_loss(self, outputs, labels, target_labels):
        loss=None
        if self.target:
            batch_size_cur = outputs.shape[0]
            labels_onehot = torch.zeros(batch_size_cur, 1000, device=self.device)
            labels_onehot.scatter_(1, target_labels.unsqueeze(1), 1)
            labels_true_onehot = torch.zeros(batch_size_cur, 1000, device=self.device)
            labels_true_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, target_labels.unsqueeze(1), float('inf'))
            loss_po = Poincare_dis(outputs / torch.sum(torch.abs(outputs), 1, keepdim=True),torch.clamp((labels_onehot - 0.00001), 0.0, 1.0))
            loss_cos = torch.clamp(Cos_dis(labels_onehot, outputs) - Cos_dis(labels_true_onehot, outputs) + 0.007, 0.0, 2.1)
            loss=loss_po + 0.01 * loss_cos
        else:
            raise Exception('Untarget attack not supported in po_trip loss.')

        return loss

    def forward(self, image, label, target_labels):
        #Logit
        image, label=image.to(self.device), label.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        batchsize = image.shape[0]
        delta = torch.zeros_like(image,requires_grad=True).to(self.device)
        grad_pre = 0
        prev = float('inf')
        for t in range(self.steps):
            outputs = self.net(self.input_diversity(image + delta))

            if self.loss=='logits':
                loss = self.logits_loss(outputs, label, target_labels)
            elif self.loss=="ce":
                loss = self.ce_loss(outputs, label, target_labels)
            if self.loss=='po_trip':
                loss = self.po_trip_loss(outputs, label, target_labels)
            loss.backward()
            grad_c = delta.grad.clone()
            grad_c = F.conv2d(grad_c, self.gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3) #TI
            grad_a = grad_c + 1 * grad_pre #MI
            grad_pre = grad_a            
            delta.grad.zero_()
            delta.data = delta.data - self.stepsize * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            delta.data = ((image + delta.data).clamp(0,1)) - image
        advimage = image+delta.data
        return advimage
    
    def input_diversity(self, x):
        img_size = x.shape[-1]#最后一个维度的值，32
        img_resize = int(img_size * self.resize_rate)#int（32*0.9）=28
        
        if self.resize_rate < 1:
            img_size = img_resize#28
            img_resize = x.shape[-1]#32
            
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)#随机生成28到32之间的整数
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x
