import torch
import numpy as np
from torchvision.transforms.functional import normalize
from ares.utils.loss import loss_adv
from ares.utils.registry import registry

@registry.register_attack('vmi_fgsm')
class VMI_fgsm(object):
    '''Enhancing the Transferability of Adversarial Attacks through Variance Tuning.

    Example:
        >>> from ares.utils.registry import registry
        >>> attacker_cls = registry.get_attack('vmi_fgsm')
        >>> attacker = attacker_cls(model)
        >>> adv_images = attacker(images, labels, target_labels)

    - Supported distance metric: 1, 2, np.inf.
    - References: https://arxiv.org/abs/2103.15571.
    '''
    def __init__(self, model, device='cuda', norm=np.inf, eps=4/255, stepsize=1/255, steps=20, decay_factor=1.0,
                 beta=1.5, sample_number=10, loss='ce', target=False):
        '''The initialize function for VMI_FGSM.

        Args:
            model (torch.nn.Module): The target model to be attacked.
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            norm (float): The norm of distance calculation for adversarial constraint. Defaults to np.inf.
            eps (float): The maximum perturbation range epsilon.
            stepsize (float): The attack range for each step.
            steps (int): The number of attack iteration.
            decay_factor (float): The decay factor.
            beta (float): The beta param.
            sample_number (int): The number of samples.
            loss (str): The loss function.
            target (bool): Conduct target/untarget attack. Defaults to False.
        '''
        self.epsilon = eps
        self.p = norm
        self.beta = beta
        self.sample_number = sample_number
        self.net = model
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
        momentum = torch.zeros_like(images).detach()
        variance = torch.zeros_like(images).detach()
        # PGD to get adversarial example

        for i in range(self.steps):
            advimage = advimage.clone().detach().requires_grad_(True)    # clone the advimage as the next iteration input

            
            netOut = self.net(advimage)
            loss = loss_adv(self.loss, netOut, labels, target_labels, self.target, self.device)      
            gradpast = torch.autograd.grad(loss, [advimage])[0].detach()
            grad = momentum * self.decay_factor + (gradpast + variance) / torch.norm(gradpast + variance, p=1)

            #update variance
            sample = advimage.clone().detach()
            global_grad = torch.zeros_like(images).detach()
            for j in range(self.sample_number):
                sample = sample.detach()
                sample.requires_grad = True
                randn = (torch.rand_like(images) * 2 - 1) * self.beta * self.epsilon
                sample = sample + randn
                outputs_sample = self.net(sample)
                loss = loss_adv(self.loss, outputs_sample, labels, target_labels, self.target, self.device) 
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
            
            advimage = torch.clamp(advimage, 0, 1)#cifar10(-1,1)
           
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
        momentum = torch.zeros_like(images).detach()
        variance = torch.zeros_like(images).detach()
        # PGD to get adversarial example

        for i in range(self.steps):
            # clone the advimages as the next iteration input
            advimages = advimages.clone().detach().requires_grad_(True)

            # normalize adversarial images for detector inputs
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
            gradpast = advimages.grad.detach()
            grad = momentum * self.decay_factor + (gradpast + variance) / torch.norm(gradpast + variance, p=1)

            # update variance
            samples = advimages.clone().detach()
            global_grad = torch.zeros_like(images).detach()
            for j in range(self.sample_number):
                samples = samples.detach()
                samples.requires_grad = True
                randn = (torch.rand_like(images) * 2 - 1) * self.beta * self.epsilon
                samples = samples + randn
                # normalize adversarial images for detector inputs
                normed_samples = normalize(samples * scale_factor, self.net.data_preprocessor.mean,
                                           self.net.data_preprocessor.std)
                losses = self.net.loss(normed_samples, batch_data['data_samples'])
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
                global_grad += torch.autograd.grad(loss, samples, grad_outputs=None, only_inputs=True)[0]
            variance = global_grad / (self.sample_number * 1.0) - gradpast

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