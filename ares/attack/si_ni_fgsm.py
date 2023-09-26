import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms.functional import normalize
from ares.utils.loss import loss_adv
from ares.utils.registry import registry

@registry.register_attack('si_ni_fgsm')
class SI_NI_FGSM(object):
    ''' Nesterov Accelerated Gradient and Scale Invariance with FGSM. A black-box attack method.

    Example:
        >>> from ares.utils.registry import registry
        >>> attacker_cls = registry.get_attack('si_ni_fgsm')
        >>> attacker = attacker_cls(model)
        >>> adv_images = attacker(images, labels, target_labels)

    - Supported distance metric: 1, 2, np.inf.
    - References: https://arxiv.org/abs/1908.06281.
    '''

    def __init__(self, model, device='cuda', norm=np.inf, eps=4/255, stepsize=1/255, steps=20,
                 scale_factor=1, decay_factor=1.0, loss='ce', target=False):
        '''The initialize function for PGD.

        Args:
            model (torch.nn.Module): The target model to be attacked.
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            norm (float): The norm of distance calculation for adversarial constraint. Defaults to np.inf.
            eps (float): The maximum perturbation range epsilon.
            stepsize (float): The attack range for each step.
            steps (int): The number of attack iteration.
            scale_factor (float): The scale factor.
            decay_factor (float): The decay factor.
            loss (str): The loss function.
            target (bool): Conduct target/untarget attack. Defaults to False.
        '''
        self.epsilon = eps
        self.p = norm
        self.net = model
        self.scale_factor = scale_factor
        self.decay_factor = decay_factor
        self.stepsize = stepsize
        self.target = target
        self.steps = steps
        self.loss = loss
        self.device = device
    
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
        batchsize = images.shape[0]
        advimage = images
        # PGD to get adversarial example
        momentum = torch.zeros_like(images).detach()
        for i in range(self.steps):
            advimage_nes = advimage + self.decay_factor * self.stepsize * momentum
            grads = torch.zeros_like(images).to(self.device)
            for j in range(self.scale_factor):
                x_s = (advimage_nes / 2**(j)).requires_grad_(True)
                netOut = self.net(x_s)
                loss = loss_adv(self.loss, netOut, labels, target_labels, self.target, self.device)     
                loss.backward(retain_graph=True)
                grads += torch.autograd.grad(loss, [x_s])[0].detach()

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
            
            advimage= torch.clamp(advimage, 0, 1)#cifar10(-1,1)
           
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
        advimages = images
        # PGD to get adversarial example
        momentum = torch.zeros_like(images).detach()
        for i in range(self.steps):
            advimage_nes = advimages + self.decay_factor * self.stepsize * momentum
            grads = torch.zeros_like(images).to(self.device)
            for j in range(self.scale_factor):
                x_s = (advimage_nes / 2 ** (j)).requires_grad_(True)
                # normalize images for detector inputs
                normed_x_s = normalize(x_s * scale_factor, self.net.data_preprocessor.mean,
                                         self.net.data_preprocessor.std)
                losses = self.net.loss(normed_x_s, batch_data['data_samples'])
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
                loss.backward(retain_graph=True)
                grads += x_s.grad.detach()

            grads_norm = torch.norm(nn.Flatten()(grads), p=1, dim=1)
            grads = grads / grads_norm.view([-1] + [1] * (len(grads.shape) - 1))
            grads = self.decay_factor * momentum + grads
            momentum = grads

            if self.p == np.inf:
                updates = grads.sign()
            else:
                normVal = torch.norm(grads.view(batchsize, -1), self.p, 1)
                updates = grads / normVal.view(batchsize, 1, 1, 1)
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