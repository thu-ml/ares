import torch
import numpy as np
from ares.utils.registry import registry

@registry.register_attack('deepfool')
class DeepFool(object):
    ''' DeepFool. A white-box iterative optimization method. It needs to calculate the Jacobian of the logits with
    relate to input, so that it only applies to tasks with small number of classification class.

    Example:
        >>> from ares.utils.registry import registry
        >>> attacker_cls = registry.get_attack('deepfool')
        >>> attacker = attacker_cls(model)
        >>> adv_images = attacker(images, labels)

    - Supported distance metric: 2, np.inf.
    - References: https://arxiv.org/abs/1511.04599.
    '''
    def __init__(self, model, device='cuda', norm=np.inf, overshoot=0.02, max_iter=50, target=False):
        '''
        Args:
            model (torch.nn.Module): The target model to be attacked.
            device  (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            norm (float): The norm of distance calculation for adversarial constraint. Defaults to np.inf.
            overshoot (float): The parameter overshoot. Defaults to 0.02.
            max_iter (int): The maximum iteration.
            target (bool): Conduct target/untarget attack. Defaults to False.
        '''

        self.overshoot = overshoot
        self.max_iter = max_iter
        self.net = model
        self.norm = norm
        self.device = device
        self.target = target
        self.min_value = 0
        self.max_value = 1
        if self.target:
            raise AssertionError('DeepFool dont support targeted attack')

    def deepfool(self, x, y):
        '''The function for deepfool.'''
        with torch.no_grad():
            logits = self.net(x)
            outputs = torch.argmax(logits, dim=1)
            if outputs!=y:
                return x
        self.nb_classes = logits.size(-1)

        adv_x = x.clone().detach().requires_grad_()
        
        iteration = 0
        logits = self.net(adv_x)
        current = logits.max(1)[1].item()
        original = logits.max(1)[1].item()
        noise = torch.zeros(x.size()).to(self.device)
        w = torch.zeros(x.size()).to(self.device)
        
        while (current == original and iteration < self.max_iter):
            gradients_0 = torch.autograd.grad(logits[0, current], [adv_x],retain_graph=True)[0].detach()
            
            for k in range(self.nb_classes):
                pert = np.inf
                if k==current:
                    continue
                gradients_1 = torch.autograd.grad(logits[0, k], [adv_x],retain_graph=True)[0].detach()
                w_k = gradients_1 - gradients_0
                f_k = logits[0, k] - logits[0, current]
                if self.norm == np.inf:
                    pert_k = (torch.abs(f_k) + 0.00001)/ torch.norm(w_k.flatten(1),1, -1)
                elif self.norm == 2:
                    pert_k = (torch.abs(f_k) + 0.00001) / torch.norm(w_k.flatten(1),2,-1)
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
                if self.norm == np.inf:
                    r_i = (pert + 1e-4) * w.sign()
                elif self.norm==2:
                    r_i = (pert + 1e-4) * w / torch.norm(w.flatten(1),2,-1)
            noise += r_i.clone()
            adv_x = x.clone().detach().requires_grad_()
            adv_x = torch.clamp(x + noise, self.min_value, self.max_value).requires_grad_()
            logits = self.net(adv_x)
            current = logits.max(1)[1].item()
            iteration = iteration + 1

        adv_x = torch.clamp((1 + self.overshoot) * noise + x, self.min_value, self.max_value)
        
        return adv_x

    def __call__(self, images=None, labels=None, target_labels=None):
        '''This function perform attack on target images with corresponding labels 
        and target labels for target attack.

        Args:
            images (torch.Tensor): The images to be attacked. The images should be torch.Tensor with shape [N, C, H, W] and range [0, 1].
            labels (torch.Tensor): The corresponding labels of the images. The labels should be torch.Tensor with shape [N, ]
            target_labels (torch.Tensor): Not used in deepfool and should be None type.

        Returns:
            torch.Tensor: Adversarial images with value range [0,1].

        '''
        assert target_labels is None, "Target attack is not supported for deepfool."

        adv_images = []
        for i in range(len(images)):
            adv_x = self.deepfool(images[i].unsqueeze(0), labels[i].unsqueeze(0))  
            adv_images.append(adv_x)
        adv_images = torch.cat(adv_images, 0)
        return adv_images