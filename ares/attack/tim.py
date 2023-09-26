import numpy as np
from scipy import stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from ares.utils.loss import loss_adv
from ares.utils.registry import registry

@registry.register_attack('tim')
class TIFGSM(object):
    '''Translation invariant attacks.

    Example:
        >>> from ares.utils.registry import registry
        >>> attacker_cls = registry.get_attack('tim')
        >>> attacker = attacker_cls(model)
        >>> adv_images = attacker(images, labels, target_labels)

    - Supported distance metric: 1, 2, np.inf.
    - References: https://arxiv.org/abs/1904.02884.
    '''
    def __init__(self, model, device='cuda', norm=np.inf, eps=4/255, stepsize=1/255, steps=20, kernel_name='gaussian', len_kernel=15, 
                 nsig=3, decay_factor=1.0, resize_rate=0.85, diversity_prob=0.7, loss='ce', target=False):
        '''The initialize function for TIFGSM.

        Args:
            model (torch.nn.Module): The target model to be attacked.
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            norm (float): The norm of distance calculation for adversarial constraint. Defaults to np.inf.
            eps (float): The maximum perturbation range epsilon.
            stepsize (float): The attack range for each step.
            steps (int): The number of attack iteration.
            kernel_name (str): The name of the kernel.
            len_kernel (int): The size for gaussian kernel.
            nsig (float): The sigma for gaussian kernel.
            decay_factor (float): The decay factor.
            resize_rate (float): The resize rate for input transform.
            diversity_prob (float): The probability of input transform.
            loss (str): The loss function.
            target (bool): Conduct target/untarget attack. Defaults to True.
        '''
        self.net = model
        self.epsilon = eps
        self.p = norm
        self.steps = steps
        self.decay = decay_factor
        self.target = target
        self.loss = loss
        self.stepsize = stepsize
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.device = device
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())

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
        images_orig = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if target_labels is not None:
            target_labels = target_labels.clone().detach().to(self.device)
        batchsize = images.shape[0]
        momentum = torch.zeros_like(images_orig).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        # random start
        delta = torch.rand_like(images)*2*self.epsilon-self.epsilon
        if self.p!=np.inf: # projected into feasible set if needed
            normVal = torch.norm(delta.view(batchsize, -1), self.p, 1)#求范数
            mask = normVal<=self.epsilon
            scaling = self.epsilon/normVal
            scaling[mask] = 1
            delta = delta*scaling.view(batchsize, 1, 1, 1)
        advimage = images+delta
        
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

    def attack_detection_forward(self, batch_data, excluded_losses, scale_factor=255.0,
                                 object_vanish_only=False):
        """This function is used to attack object detection models.

        Args:
            batch_data (dict): {'inputs': torch.Tensor with shape [N,C,H,W] and value range [0, 1], 'data_samples': list of mmdet.structures.DetDataSample}.
            excluded_losses (list): List of losses not used to compute the attack loss.
            scale_factor (float): Factor used to scale adv images.
            object_vanish_only (bool): When True, just make objects vanish only.

        Returns:
            torch.Tensor: Adversarial images with value range [0,1].

        """

        images = batch_data['inputs']
        batchsize = len(images)

        images_orig = images.clone().detach().to(self.device)
        momentum = torch.zeros_like(images_orig).detach().to(self.device)
        stacked_kernel = self.stacked_kernel.to(self.device)

        # random start
        delta = torch.rand_like(images) * 2 * self.epsilon - self.epsilon
        if self.p != np.inf:  # projected into feasible set if needed
            normVal = torch.norm(delta.view(batchsize, -1), self.p, 1)  # 求范数
            mask = normVal <= self.epsilon
            scaling = self.epsilon / normVal
            scaling[mask] = 1
            delta = delta * scaling.view(batchsize, 1, 1, 1)
        advimages = images + delta

        for i in range(self.steps):
            advimages = advimages.clone().detach().requires_grad_(True)

            normed_advimages = normalize(advimages * scale_factor, self.net.data_preprocessor.mean,
                                         self.net.data_preprocessor.std)
            losses = self.net.loss(normed_advimages, batch_data['data_samples'])
            loss = []
            for key in losses.keys():
                if isinstance(losses[key], list):
                    losses[key] = torch.stack(losses[key]).mean()
                kept = True
                for excluded_loss in excluded_losses:
                    if excluded_loss in key:
                        kept = False
                        continue
                if kept and 'loss' in key:
                    loss.append(losses[key].mean().unsqueeze(0))
            if object_vanish_only:
                loss = - torch.stack(loss).mean()
            else:
                loss = torch.stack((loss)).mean()
            advimages.grad = None
            loss.backward()
            grad = advimages.grad.detach()
            # depth wise conv2d
            grad = F.conv2d(grad, stacked_kernel, stride=1, padding=int((self.len_kernel - 1) / 2), groups=3)
            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            grad = grad / grad_norm.view([-1] + [1] * (len(grad.shape) - 1))
            grad = grad + momentum * self.decay
            momentum = grad

            if self.p == np.inf:
                updates = grad.sign()
            else:
                normVal = torch.norm(grad.view(batchsize, -1), self.p, 1)
                updates = grad / normVal.view(batchsize, 1, 1, 1)
            updates = updates * self.stepsize
            advimages = advimages + updates
            # project the disturbed image to feasible set if needed
            delta = advimages - images
            if self.p == np.inf:
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            else:
                normVal = torch.norm(delta.view(batchsize, -1), self.p, 1)
                mask = normVal <= self.epsilon
                scaling = self.epsilon / normVal
                scaling[mask] = 1
                delta = delta * scaling.view(batchsize, 1, 1, 1)
            advimages = images + delta

            advimages = torch.clamp(advimages, 0, 1)

        return advimages

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