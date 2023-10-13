import torch
from attacks.utils import *
from torch import nn
from typing import Callable, List
from .AdversarialInputBase import AdversarialInputAttacker

__all__ = ['NAttack']


class NAttack(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module],
                 total_step: int = 600,
                 step_size: float = 0.008,
                 batch_size=300,
                 sigma=0.1,
                 ):
        self.total_step = total_step
        self.step_size = step_size
        self.batch_size = batch_size
        super(NAttack, self).__init__(model, *args, **kwargs)
        self.sigma = sigma

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def g(self, x):
        return (torch.tanh(x) + 1) / 2

    def proj(self, mu, noise):
        return clamp(clamp(noise, mu - self.epsilon, mu + self.epsilon))

    @torch.no_grad()
    def attack(self, x, y, ):
        assert x.shape[0] == 1, 'now only support batch size = 1'
        x.requires_grad = False
        N, C, H, D = x.shape
        original_x = x.clone()
        mu = torch.arctan(2 * x - 1)
        mu_min = torch.arctan(2 * (x - self.epsilon) - 1)
        mu_max = torch.arctan(2 * (x + self.epsilon) - 1)
        for _ in range(self.total_step):
            epsilons = torch.randn(self.batch_size, C, H, D, device=self.device)
            samples = mu + self.sigma * epsilons
            samples = self.g(samples)
            samples = self.proj(original_x, samples)
            z_scores = []
            for model in self.models:
                out = model(samples)
                pre = torch.max(out, dim=1)[1]
                mask = (pre != y).float()
                if torch.sum(mask) > 0:
                    index = torch.max(mask, dim=0)[1]
                    result = samples[index].unsqueeze(0)
                    return result
                f = out[:, y]  # batch_size
                z_score = (f - torch.mean(f)) / (torch.std(f) + 1e-7)
                z_scores.append(z_score)
            z_scores = torch.stack(z_scores).mean(0)  # batch_size
            mu = mu - self.step_size / (self.batch_size * self.sigma) * (z_scores.view(-1, 1, 1, 1) * epsilons).sum(0)
            mu = clamp(mu, min_value=mu_min, max_value=mu_max)
        result = self.proj(original_x, self.g(mu))
        return result
