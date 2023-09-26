import torch
import numpy as np
from ares.utils.registry import registry

@registry.register_attack('nes')
class NES(object):
    ''' Natural Evolution Strategies (NES). A black-box constraint-based method. Use NES as gradient estimation
    technique and employ PGD with this estimated gradient to generate the adversarial example.

    Example:
        >>> from ares.utils.registry import registry
        >>> attacker_cls = registry.get_attack('nes')
        >>> attacker = attacker_cls(model)
        >>> adv_images = attacker(images, labels, target_labels)

    - Supported distance metric: 1, 2, np.inf.
    - References:
      1. https://arxiv.org/abs/1804.08598.
      2. http://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf.
    '''
    def __init__(self, model, device='cuda', norm=np.inf, eps=4/255, stepsize=1/255, nes_samples=10, sample_per_draw=1, 
                 max_queries=1000, search_sigma=0.02, decay=0.00, random_perturb_start=False, target=False):
        '''The initialize function for NES.

        Args:
            model (torch.nn.Module): The target model to be attacked.
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            norm (float): The norm of distance calculation for adversarial constraint. Defaults to np.inf.
            eps (float): The maximum perturbation range epsilon.
            stepsize (float): The step size for each attack iteration. Defaults to 1/255.
            nes_samples (int): The samples for NES.
            sample_per_draw (int): Sample in each draw.
            max_queries (int): Maximum query number.
            search_sigma (float): The sigma param for searching.
            decay (float): Decay rate.
            random_perturb_start (bool): Whether start with random perturbation.
            target (bool): Conduct target/untarget attack. Defaults to False.
        '''
        self.model = model
        self.p = norm
        self.epsilon = eps
        self.step_size = stepsize
        self.max_queries = max_queries
        self.device = device
        self.search_sigma = search_sigma
        nes_samples = nes_samples if nes_samples else sample_per_draw
        self.nes_samples = (nes_samples // 2) *2
        self.sample_per_draw = (sample_per_draw // self.nes_samples) * self.nes_samples
        self.nes_iters = self.sample_per_draw // self.nes_samples
        self.decay = decay
        self.random_perturb_start = random_perturb_start 
        self.target = target
        
        self.min_value = 0
        self.max_value = 1
    
    def _is_adversarial(self,x, y, y_target):
        '''The function to judge if the input image is adversarial.'''
        output = torch.argmax(self.model(x), dim=1)
        if self.target:
            return output == y_target
        else:
            return output != y
    

    def _margin_logit_loss(self, x, labels, target_labels):
        '''The function to calculate the marginal logits.'''
        outputs = self.model(x)
        if self.target:
            one_hot_labels = torch.eye(len(outputs[0]))[target_labels].to(self.device)
            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            cost = -torch.clamp((i-j), min=0)     # -self.kappa=0
        else:
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)
            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            cost = -torch.clamp((j-i), min=0)     # -self.kappa=0
        return cost
        
    def clip_eta(self, batchsize, eta, norm, eps):
        '''The function to clip image according to the constraint.'''
        if norm == np.inf:
            eta = torch.clamp(eta, -eps, eps)
        elif norm == 2:
            normVal = torch.norm(eta.view(batchsize, -1), self.p, 1)
            mask = normVal<=eps
            scaling = eps/normVal
            scaling[mask] = 1
            eta = eta*scaling.view(batchsize, 1, 1, 1)
        else:
            raise NotImplementedError
        return eta

    def nes_gradient(self, x, y, ytarget):
        '''The function to calculate the gradient of NES.'''
        x_shape = x.size()
        g = torch.zeros(x_shape).to(self.device)
        mean = torch.zeros(x_shape).to(self.device)
        std = torch.ones(x_shape).to(self.device)

        for i in range(self.nes_iters):
            u = torch.normal(mean, std).to(self.device)
            pred = self._margin_logit_loss(torch.clamp(x+self.search_sigma*u,self.min_value,self.max_value),y,ytarget)
            g = g + pred*u
            pred = self._margin_logit_loss(torch.clamp(x-self.search_sigma*u,self.min_value,self.max_value), y,ytarget)
            g = g - pred*u

        return g/(2*self.nes_iters*self.search_sigma)
    

    def nes(self, x_victim, y_victim, y_target):
        '''The attack process of NES.'''
        batchsize = x_victim.shape[0]
        with torch.no_grad():
            self.model.eval()
            x_victim = x_victim.to(self.device)
            y_victim = y_victim.to(self.device)
            if y_target is not None:
                y_target = y_target.to(self.device)
            self.model.to(self.device)
           
            if self._is_adversarial(x_victim, y_victim, y_target):
                self.detail['queries'] = 0
                self.detail['success'] = True
                return x_victim
            
            self.detail['success'] = False
            queries = 0
            
            x_adv = x_victim.clone().to(self.device)

            if self.random_perturb_start:
                noise = torch.rand(x_adv.size()).to(self.device)
                normalized_noise = self.clip_eta(batchsize, noise, self.p, self.epsilon)
                x_adv += normalized_noise

            momentum = torch.zeros_like(x_adv)
            self.model.eval()

            while queries+self.sample_per_draw <=  self.max_queries:
                queries += self.sample_per_draw
                x_adv.requires_grad = True
                self.model.zero_grad()
                grad = self.nes_gradient(x_adv, y_victim, y_target)
                grad = grad + momentum * self.decay
                momentum = grad
                if self.p==np.inf:
                    updates = grad.sign()
                else:
                    normVal = torch.norm(grad.view(batchsize, -1), self.p, 1)
                    updates = grad/normVal.view(batchsize, 1, 1, 1)
                updates = updates*self.step_size
                
                x_adv = x_adv + updates
                delta = x_adv-x_victim
                delta = self.clip_eta(batchsize, delta, self.p, self.epsilon)
                x_adv = torch.clamp(x_victim + delta, min=self.min_value, max=self.max_value).detach()
                
                if self._is_adversarial(x_adv, y_victim, y_target):
                    self.detail['success'] = True
                    break
            self.detail['queries'] = queries
            return x_adv

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
        adv_images = []
        self.detail = {}
        for i in range(len(images)):
            if target_labels is None:
                target_label = None
            else:
                target_label = target_labels[i].unsqueeze(0)
            adv_x = self.nes(images[i].unsqueeze(0), labels[i].unsqueeze(0), target_label)
            adv_images.append(adv_x)
        adv_images = torch.cat(adv_images, 0)
        return adv_images
            
            