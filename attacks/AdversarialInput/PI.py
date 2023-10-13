import torch
from attacks.utils import *
from torch import nn
from typing import Callable, List
from .AdversarialInputBase import AdversarialInputAttacker
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable as V

"""
reference:
https://github.com/qilong-zhang/Patch-wise-iterative-attack
"""


class PI_FGSM(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 amplification_factor: float = 5,
                 k: int = 15,
                 *args, **kwargs
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(PI_FGSM, self).__init__(model, *args, **kwargs)
        self.amplification_factor = amplification_factor
        self.gamma = self.epsilon / self.total_step * amplification_factor
        self.step_size = self.epsilon / self.total_step
        self.kern = self.project_kern(3)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    # def attack(self, x, y, ):
    #     N = x.shape[0]
    #     original_x = x.clone()
    #     a = torch.zeros_like(x)
    #     momentum = torch.zeros_like(x)
    #     if self.random_start:
    #         x = self.perturb(x)
    #
    #     for _ in range(self.total_step):
    #         x.requires_grad = True
    #         logit = 0
    #         for model in self.models:
    #             logit += model(x.to(model.device)).to(x.device)
    #         loss = self.criterion(logit, y)
    #         loss.backward()
    #         grad = x.grad
    #         x.requires_grad = False
    #         # update
    #         if self.targerted_attack:
    #             momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
    #             a = self.mu * a - grad.sign() * self.amplification_factor * self.step_size
    #         else:
    #             momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
    #             a = self.mu * a + grad.sign() * self.amplification_factor * self.step_size
    #         momentum_norm = torch.max(torch.abs(a.view(N, -1)), dim=1)[0]
    #         mask = momentum_norm > self.epsilon
    #         if torch.sum(mask) > 0:
    #             c = clamp(torch.abs(a[mask]) - self.epsilon, min_value=0, max_value=float('inf'))
    #             c = c * torch.sign(a[mask])
    #             a[mask] = a[mask] + self.gamma * torch.sign(self.project_noise(c, *self.kern))
    #             x = x + self.amplification_factor * self.step_size * momentum.sign()
    #             x[mask] = x[mask] + self.gamma * torch.sign(self.project_noise(c, *self.kern))
    #         else:
    #             x = x + self.amplification_factor * self.step_size * momentum.sign()
    #         x = self.clamp(x, original_x)
    #     return x

    @staticmethod
    def project_kern(kern_size):
        kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
        kern[kern_size // 2, kern_size // 2] = 0.0
        kern = kern.astype(np.float32)
        stack_kern = np.stack([kern, kern, kern])
        stack_kern = np.expand_dims(stack_kern, 1)
        stack_kern = torch.tensor(stack_kern).cuda()
        return stack_kern, kern_size // 2

    @staticmethod
    def project_noise(x, stack_kern, padding_size):
        x = F.conv2d(x, stack_kern, padding=(padding_size, padding_size), groups=3)
        return x

    def attack(self, x, gt, x_min=0, x_max=1, num_iter=10, ):
        amplification = self.amplification_factor
        # x_min = torch.clamp(x - self.epsilon, 0.0, 1.0)
        # x_max = torch.clamp(x + self.epsilon, 0.0, 1.0)
        ori_x = x.clone()
        eps = self.epsilon
        alpha = eps / num_iter
        alpha_beta = alpha * amplification
        gamma = alpha_beta
        x.requires_grad = True
        amplification = 0.0
        momentum = torch.zeros_like(x)
        for i in range(num_iter):
            x.grad = None
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, gt)
            loss.backward()
            noise = x.grad.data

            # MI-FGSM
            noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
            grad = noise
            noise = momentum * grad + noise

            amplification += alpha_beta * torch.sign(noise)
            cut_noise = torch.clamp(torch.abs(amplification) - eps, 0, 10000.0) * torch.sign(amplification)
            projection = gamma * torch.sign(self.project_noise(cut_noise, *self.kern))
            amplification += projection

            # x = x + alpha * torch.sign(noise)
            x = x + alpha_beta * torch.sign(noise) + projection
            x = self.clamp(x, ori_x)
            x = torch.clamp(x, min=x_min, max=x_max)
            x = V(x, requires_grad=True)

        return x.detach()
