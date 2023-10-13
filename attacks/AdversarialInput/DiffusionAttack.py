import torch
from attacks.utils import *
from torch import nn
from typing import Callable, List
from .AdversarialInputBase import AdversarialInputAttacker
from torchvision import transforms
from .utils import cosine_similarity


class DiffusionAttack(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 10,
                 criterion: Callable = nn.MSELoss(),
                 *args, **kwargs
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        super(DiffusionAttack, self).__init__(model, *args, **kwargs)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y=None):
        original_x = x.clone()
        grads = []
        if self.random_start:
            x = self.perturb(x)

        for step in range(1, self.total_step + 1):
            x.requires_grad = True
            model = self.models[0]
            purified = model(x)
            loss = self.criterion(purified, x) * 1e5
            print(loss)
            loss.backward()
            grad = x.grad
            # print(grad)
            grads.append(grad)
            x.requires_grad = False
            x += self.step_size * grad.sign()
            x = self.clamp(x, original_x)
        diff = torch.abs(original_x - x)
        logic = (torch.abs(diff - self.epsilon) < 1 / 255).float() + (x < 1 / 255).float() + ((1 - x) < 1 / 255).float()
        print(torch.sum(torch.clamp(logic, max=1)) / diff.numel())
        print(cosine_similarity(grads))
        return x


class UNetAttack(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 10,
                 criterion: Callable = nn.MSELoss(),
                 *args, **kwargs
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        super(UNetAttack, self).__init__(model, *args, **kwargs)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y=None, max_t=10):
        B = x.shape[0]
        original_x = x.clone()
        grads = []
        if self.random_start:
            x = self.perturb(x)

        for step in range(1, self.total_step + 1):
            x.requires_grad = True
            model = self.models[0]
            transformed = ((x - 0.5) * 2).expand(max_t * B, *x.shape[1:])  # max_t*B
            tensor_t = torch.arange(max_t).unsqueeze(1).expand(max_t, B).view(-1).to(x.device)  # max_t*B
            pre = model(transformed, tensor_t)
            loss = torch.norm(pre.view(pre.shape[0], -1), p=2, dim=1).mean()
            print(loss)
            loss.backward()
            grad = x.grad
            # print(grad)
            grads.append(grad)
            x.requires_grad = False
            x += self.step_size * grad.sign()
            x = self.clamp(x, original_x)
        diff = torch.abs(original_x - x)
        logic = (torch.abs(diff - self.epsilon) < 1 / 255).float() + (x < 1 / 255).float() + ((1 - x) < 1 / 255).float()
        print(torch.sum(torch.clamp(logic, max=1)) / diff.numel())
        print(cosine_similarity(grads))
        return x
