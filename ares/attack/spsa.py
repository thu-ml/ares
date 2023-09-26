import numpy as np
import torch
from torch import optim
from ares.utils.registry import registry

@registry.register_attack('spsa')
class SPSA(object):
    ''' Simultaneous Perturbation Stochastic Approximation (SPSA). A black-box constraint-based method. Use SPSA as
    gradient estimation technique and employ Adam with this estimated gradient to generate the adversarial example.

    Example:
        >>> from ares.utils.registry import registry
        >>> attacker_cls = registry.get_attack('spsa')
        >>> attacker = attacker_cls(model)
        >>> adv_images = attacker(images, labels, target_labels)

    - Supported distance metric: 1, 2, np.inf.
    - References: https://arxiv.org/abs/1802.05666.
    '''
    def __init__(self, model, device='cuda', norm=np.inf, eps=4/255, learning_rate=0.01, delta=0.01, spsa_samples=10,
                 sample_per_draw=1, nb_iter=20, early_stop_loss_threshold=None, target=False):
        '''The initialize function for SPSA.

        Args:
            model (torch.nn.Module): The target model to be attacked.
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            norm (float): The norm of distance calculation for adversarial constraint. Defaults to np.inf.
            eps (float): The maximum perturbation range epsilon.
            learning_rate (float): The learning rate of attack.
            delta (float): The delta param.
            spsa_samples (int): Number of samples in SPSA.
            sample_per_draw (int): Sample in each draw.
            nb_iter (int): Number of iteration.
            early_stop_loss_threshold (float): The threshold for early stop.
            target (bool): Conduct target/untarget attack. Defaults to False.
        '''
        self.model = model
        self.device = device
        self.IsTargeted = target
        self.eps = eps
        self.learning_rate = learning_rate
        self.delta = delta
        spsa_samples = spsa_samples if spsa_samples else sample_per_draw
        self.spsa_samples = (spsa_samples // 2) *2
        self.sample_per_draw = (sample_per_draw // self.spsa_samples) * self.spsa_samples
        assert self.sample_per_draw > 0, "Sample per draw must not be zero."
        self.nb_iter = nb_iter
        self.norm = norm
        self.early_stop_loss_threshold = early_stop_loss_threshold
        self.clip_min = 0
        self.clip_max = 1    

    def clip_eta(self, batchsize, eta, norm, eps):
        '''The function to clip image according to the constraint.'''
        if norm == np.inf:
            eta = torch.clamp(eta, -eps, eps)
        elif norm == 2:
            normVal = torch.norm(eta.view(batchsize, -1), self.norm, 1)
            mask = normVal<=eps
            scaling = eps/normVal
            scaling[mask] = 1
            eta = eta*scaling.view(batchsize, 1, 1, 1)
        else:
            raise NotImplementedError
        return eta
   
   
    def _get_batch_sizes(self, n, max_batch_size):
        batches = [max_batch_size for _ in range(n // max_batch_size)]
        if n % max_batch_size > 0:
            batches.append(n % max_batch_size)
        return batches
    
    def _compute_spsa_gradient(self, loss_fn, x, delta, nb_sample, max_batch_size):
        '''The function to calculate the gradient of SPSA.'''
        grad = torch.zeros_like(x)
        x = x.unsqueeze(0)
        x = x.expand(max_batch_size, *x.shape[1:]).contiguous()
        v = torch.empty_like(x[:, :1, ...])

        for batch_size in self._get_batch_sizes(nb_sample, max_batch_size):
            x_ = x[:batch_size]
            vb = v[:batch_size]
            vb = vb.bernoulli_().mul_(2.0).sub_(1.0)
            v_ = vb.expand_as(x_).contiguous()
            x_shape = x_.shape
            x_ = x_.view(-1, *x.shape[2:])
            v_ = v_.view(-1, *v.shape[2:])
            df = loss_fn(delta * v_) - loss_fn(- delta * v_)
            df = df.view(-1, *[1 for _ in v_.shape[1:]])
            grad_ = df / (2. * delta * v_)
            grad_ = grad_.view(x_shape)
            grad_ = grad_.sum(dim=0, keepdim=False)
            grad += grad_
        grad /= nb_sample

        return grad
        
    
    def _is_adversarial(self,x, y, y_target):
        '''The function to judge if the input image is adversarial.'''
        output = torch.argmax(self.model(x), dim=1)
        if self.IsTargeted:
            return output == y_target
        else:
            return output != y
    
    
    def _margin_logit_loss(self, logits, labels, target_label):
        '''The function to calculate the marginal logits.'''
        if self.IsTargeted:
            correct_logits = logits.gather(1, target_label[:, None]).squeeze(1)

            logit_indices = torch.arange(logits.size()[1], dtype=target_label.dtype, device=target_label.device)[None, :].expand(target_label.size()[0], -1)
            incorrect_logits = torch.where(logit_indices == target_label[:, None], torch.full_like(logits, float("-inf")), logits)
            max_incorrect_logits, _ = torch.max(incorrect_logits, 1)

            return max_incorrect_logits -correct_logits
        else:
            correct_logits = logits.gather(1, labels[:, None]).squeeze(1)

            logit_indices = torch.arange(logits.size()[1], dtype=labels.dtype, device=labels.device)[None, :].expand(labels.size()[0], -1)
            incorrect_logits = torch.where(logit_indices == labels[:, None], torch.full_like(logits, float("-inf")), logits)
            max_incorrect_logits, _ = torch.max(incorrect_logits, 1)

            return -(max_incorrect_logits-correct_logits)

    def spsa(self,x, y, y_target):
        '''The main function of SPSA attack.'''
        device = self.device
        eps = self.eps
        batchsize = x.shape[0]
        learning_rate = self.learning_rate
        delta = self.delta
        spsa_samples = self.spsa_samples
        nb_iter = self.nb_iter

        v_x = x.to(device)
        v_y = y.to(device)

        if self._is_adversarial(v_x, v_y, y_target):
            self.detail['queries'] = 0
            self.detail['success'] = True
            return v_x


        perturbation = (torch.rand_like(v_x) * 2 - 1) * eps
        optimizer = optim.Adam([perturbation], lr=learning_rate)
        
        self.detail['success'] = False
        queries = 0
        
        while queries+self.sample_per_draw <= nb_iter:
            queries += self.sample_per_draw
            def loss_fn(pert):
                input1 = v_x + pert
                input1 = torch.clamp(input1, self.clip_min, self.clip_max)
                logits = self.model(input1)
                return self._margin_logit_loss(logits, v_y.expand(len(pert)), y_target.expand(len(pert))) if self.IsTargeted else self._margin_logit_loss(logits, v_y.expand(len(pert)), None)
         
            spsa_grad = self._compute_spsa_gradient(loss_fn, v_x, delta=delta, nb_sample=spsa_samples, max_batch_size=self.sample_per_draw)
            perturbation.grad = spsa_grad
            optimizer.step()

            clip_perturbation = self.clip_eta(batchsize, perturbation, self.norm, eps)
            adv_image = torch.clamp(v_x + clip_perturbation, self.clip_min, self.clip_max)
            perturbation.add_((adv_image - v_x) - perturbation)
            loss = loss_fn(perturbation).item()
   
            if (self.early_stop_loss_threshold is not None and loss < self.early_stop_loss_threshold):
                break

            if self._is_adversarial(adv_image, v_y, y_target):
                self.detail['success'] = True
                break
        self.detail['queries'] = queries
        return adv_image

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
            adv_x = self.spsa(images[i].unsqueeze(0), labels[i].unsqueeze(0), target_label)
            adv_images.append(adv_x)
        adv_images = torch.cat(adv_images, 0)
        return adv_images
