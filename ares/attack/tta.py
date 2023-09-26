import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
from ares.utils.registry import registry

# evaluate simple transferable attacks in the single-model transfer setting
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

@registry.register_attack('tta')
class TTA(object):
    '''Transferable Targeted Attacks.

    Example:
        >>> from ares.utils.registry import registry
        >>> attacker_cls = registry.get_attack('tta')
        >>> attacker = attacker_cls(model)
        >>> adv_images = attacker(images, labels, target_labels)

    - Supported distance metric: 1, 2, np.inf.
    - References: https://arxiv.org/abs/2012.11207.
    '''

    def __init__(self, model, device='cuda', norm=np.inf, eps=4/255, stepsize=1/255, steps=20, kernel_size=5,
                 nsig=3, resize_rate=0.85, diversity_prob=0.7, loss='ce', target=True):
        '''The initialize function for TTA.

        Args:
            model (torch.nn.Module): The target model to be attacked.
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            norm (float): The norm of distance calculation for adversarial constraint. Defaults to np.inf.
            eps (float): The maximum perturbation range epsilon.
            stepsize (float): The attack range for each step.
            steps (int): The number of attack iteration.
            kernel_size (int): The size for gaussian kernel.
            nsig (float): The sigma for gaussian kernel.
            resize_rate (float): The resize rate for input transform.
            diversity_prob (float): The probability of input transform.
            loss (str): The loss function.
            target (bool): Conduct target/untarget attack. Defaults to True.

        '''
        self.epsilon = eps
        self.p = norm
        self.net = model
        self.stepsize = stepsize
        self.steps = steps
        self.loss = loss
        self.target = target
        self.device = device
        self.resize_rate=resize_rate
        self.diversity_prob=diversity_prob
        self.gaussian_kernel=TI_tta(kernel_size, nsig).to(device)
    
    def ce_loss(self, outputs, labels, target_labels):
        '''Function of ce loss for TTA.'''
        loss = nn.CrossEntropyLoss()
        
        if self.target:
            cost = loss(outputs, target_labels)
        else:
            cost = -loss(outputs, labels)
        return cost


    def logits_loss(self, outputs, labels, target_labels):
        '''The logits function.'''
        loss=None
        if self.target:
            real = outputs.gather(1,target_labels.unsqueeze(1)).squeeze(1)
            logit_dists = ( -1 * real)
            loss = logit_dists.sum()
        else:
            raise Exception('Untarget attack not supported in logits loss.')
        return loss
    
    def po_trip_loss(self, outputs, labels, target_labels):
        '''The function to calculate po trip loss.'''
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
        #Logit
        images, labels=images.to(self.device), labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        batchsize = images.shape[0]
        delta = torch.zeros_like(images,requires_grad=True).to(self.device)
        grad_pre = 0
        prev = float('inf')
        for t in range(self.steps):
            outputs = self.net(self.input_diversity(images + delta))

            if self.loss=='logits':
                loss = self.logits_loss(outputs, labels, target_labels)
            elif self.loss=="ce":
                loss = self.ce_loss(outputs, labels, target_labels)
            if self.loss=='po_trip':
                loss = self.po_trip_loss(outputs, labels, target_labels)
            loss.backward()
            grad_c = delta.grad.clone()
            grad_c = F.conv2d(grad_c, self.gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3) #TI
            grad_a = grad_c + 1 * grad_pre #MI
            grad_pre = grad_a            
            delta.grad.zero_()
            delta.data = delta.data - self.stepsize * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            delta.data = ((images + delta.data).clamp(0,1)) - images
        advimage = images+delta.data
        return advimage
    
    def input_diversity(self, x):
        '''The function to perform random input transform.'''
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
