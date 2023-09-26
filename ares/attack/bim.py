import torch
import numpy as np
from ares.utils.loss import loss_adv
from torchvision.transforms.functional import normalize
from ares.utils.registry import registry

@registry.register_attack('bim')
class BIM(object):
    ''' Basic Iterative Method (BIM). A white-box iterative constraint-based method. Require a differentiable loss function.

    Example:
        >>> from ares.utils.registry import registry
        >>> attacker_cls = registry.get_attack('bim')
        >>> attacker = attacker_cls(model)
        >>> adv_images = attacker(images, labels, target_labels)

    - Supported distance metric: 1, 2, np.inf.
    - References: https://arxiv.org/abs/1607.02533.
    '''

    def __init__(self, model, device='cuda', norm=np.inf, eps=4/255, stepsize=1/255, steps=20, target=False, loss='ce'):
        '''
        Args:
            model (torch.nn.Module): The target model to be attacked.
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            norm (float): The norm of distance calculation for adversarial constraint. Defaults to np.inf. It is selected from [1, 2, np.inf].
            eps (float): The maximum perturbation range epsilon.
            stepsize (float): The step size for each attack iteration. Defaults to 1/255.
            steps (int): The attack steps. Defaults to 20.
            target (bool): Conduct target/untarget attack. Defaults to False.
            loss (str): The loss function. Defaults to 'ce'.
        '''

        self.epsilon = eps
        self.p = norm
        self.net = model
        self.stepsize = stepsize
        self.steps = steps
        self.target = target
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

        batchsize = images.shape[0]
        images, labels = images.to(self.device), labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        advimage = images
       
        

        for i in range(self.steps):
            advimage = advimage.clone().detach().requires_grad_(True).to(self.device)
            netOut = self.net(advimage)
            loss = loss_adv(self.loss, netOut, labels, target_labels, self.target, self.device) 
            updates = torch.autograd.grad(loss, [advimage])[0].detach()
            if self.p==np.inf:
                updates = updates.sign()
            else:
                normVal = torch.norm(updates.view(batchsize, -1), self.p, 1)
                updates = updates/normVal.view(batchsize, 1, 1, 1)
            updates = updates*self.stepsize
            
            advimage = advimage + updates
            
            delta = advimage - images
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
        batchsize = images.shape[0]
        advimages = images

        for i in range(self.steps):
            advimages = advimages.clone().detach().requires_grad_(True).to(self.device)

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
            updates = advimages.grad.detach()
            if self.p == np.inf:
                updates = updates.sign()
            else:
                normVal = torch.norm(updates.view(batchsize, -1), self.p, 1)
                updates = updates / normVal.view(batchsize, 1, 1, 1)
            updates = updates * self.stepsize

            advimages = advimages + updates

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