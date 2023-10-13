import torch
from attacks.utils import *
from torch import nn
from typing import Callable, List
from .AdversarialInputBase import AdversarialInputAttacker
import numpy as np
from scipy import stats as st


class MI_TI_FGSM(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 10,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(MI_TI_FGSM, self).__init__(model, *args, **kwargs)
        self.conv = self.gkern().to(self.device)
        self.conv.requires_grad_(False)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            # update
            grad = self.conv(grad)
            if self.targerted_attack:
                momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            x = clamp(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x

    @staticmethod
    def gkern(kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = torch.tensor(kernel, dtype=torch.float)
        conv = nn.Conv2d(3, 3, kernel_size=kernlen, stride=1, padding=kernlen // 2, bias=False, groups=3)
        kernel = kernel.repeat(3, 1, 1).view(3, 1, kernlen, kernlen)
        conv.weight.data = kernel
        return conv


if __name__ == '__main__':
    MI_TI_FGSM.gkern()
