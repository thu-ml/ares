import torch
from .AdversarialInputBase import AdversarialInputAttacker
from typing import Callable, List, Iterable
from attacks.utils import *
from .utils import cosine_similarity
from torch import nn
import random
from torchvision import transforms
import numpy as np
from scipy import stats as st


class MI_CosineSimilarityEncourager(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10,
                 random_start: bool = False,
                 step_size: float = 50,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu=1,
                 outer_optimizer=None,
                 *args,
                 **kwargs
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        self.outer_optimizer = outer_optimizer
        super(MI_CosineSimilarityEncourager, self).__init__(model, *args, **kwargs)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        self.outer_momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            self.begin_attack(x.clone().detach())
            for model in self.models:
                x.requires_grad = True
                loss = self.criterion(model(x.to(model.device)), y.to(model.device))
                loss.backward()
                grad = x.grad
                self.grad_record.append(grad)
                x.requires_grad = False
                # update
                if self.targerted_attack:
                    momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1)
                    x += self.step_size * momentum
                else:
                    momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1)
                    x += self.step_size * momentum
                    # x += self.step_size * grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1)
                x = clamp(x)
                x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            x = self.end_attack(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x

    @torch.no_grad()
    def begin_attack(self, origin: torch.tensor):
        self.original = origin
        self.grad_record = []

    @torch.no_grad()
    def end_attack(self, now: torch.tensor, ksi=16 / 255 / 5):
        '''
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        '''
        patch = now
        if self.outer_optimizer is None:
            fake_grad = (patch - self.original)
            self.outer_momentum = self.mu * self.outer_momentum + fake_grad / torch.norm(fake_grad, p=1)
            patch.mul_(0)
            patch.add_(self.original)
            patch.add_(ksi * self.outer_momentum.sign())
            # patch.add_(ksi * fake_grad)
        else:
            fake_grad = - ksi * (patch - self.original)
            self.outer_optimizer.zero_grad()
            patch.mul_(0)
            patch.add_(self.original)
            patch.grad = fake_grad
            self.outer_optimizer.step()
        patch = clamp(patch)
        grad_similarity = cosine_similarity(self.grad_record)
        del self.grad_record
        del self.original
        return patch


class MI_RandomWeight(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 50,
                 *args,
                 **kwargs
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(MI_RandomWeight, self).__init__(model, *args, **kwargs)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def random_by_mean(self, mean: float = 1, eps=5) -> float:
        '''
        random a number in [0, 2*mean]. The expectation is mean.
        :param mean:
        :return:
        '''
        x = (random.random() - 0.5) * 2  # with range [-1, 1], mean 0
        x *= eps  # delta = 2*eps
        x = x + mean  # expectation is mean
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            x.requires_grad = True
            # loss = 0
            # for model in self.models:
            #     loss += self.criterion(model(x.to(model.device)), y.to(model.device)).to(x.device) \
            #             * self.random_by_mean()
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device) * self.random_by_mean()
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            # update
            if self.targerted_attack:
                momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            x = clamp(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x


class MI_CommonWeakness(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10,
                 random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu=1,
                 reverse_step_size=16 / 255 / 15,
                 inner_step_size: float = 250,
                 DI=False,
                 TI=False,
                 *args,
                 **kwargs
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        self.reverse_step_size = reverse_step_size
        super(MI_CommonWeakness, self).__init__(model, *args, **kwargs)
        self.inner_step_size = inner_step_size
        self.DI = DI
        self.TI = TI
        if DI:
            self.aug_policy = transforms.Compose([
                transforms.RandomCrop((int(224 * 0.9), int(224 * 0.9)), padding=224 - int(224 * 0.9)),
            ])
        else:
            self.aug_policy = nn.Identity()
        if TI:
            self.ti = self.gkern().to(self.device)
            self.ti.requires_grad_(False)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        inner_momentum = torch.zeros_like(x)
        self.outer_momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # first step, MI-SAM
            self.begin_attack(x.clone().detach())
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad
            if self.TI:
                grad = self.ti(grad)
            x.requires_grad = False
            if self.targerted_attack:
                x += self.reverse_step_size * grad.sign()
            else:
                x -= self.reverse_step_size * grad.sign()
                # x -= self.reverse_step_size * grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1)
            x = clamp(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            # --------------------------------------------------------------------------------#
            # second step, MI-CSE
            x.grad = None
            # self.begin_attack(x.clone().detach())
            for model in self.models:
                x.requires_grad = True
                aug_x = self.aug_policy(x)
                loss = self.criterion(model(aug_x.to(model.device)), y.to(model.device))
                loss.backward()
                grad = x.grad
                self.grad_record.append(grad)
                x.requires_grad = False
                # update
                if self.TI:
                    grad = self.ti(grad)
                if self.targerted_attack:
                    inner_momentum = self.mu * inner_momentum - grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                        N, 1, 1, 1)
                    x += self.inner_step_size * inner_momentum
                else:
                    inner_momentum = self.mu * inner_momentum + grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                        N, 1, 1, 1)
                    x += self.inner_step_size * inner_momentum
                x = clamp(x)
                x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            x = self.end_attack(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x

    @torch.no_grad()
    def begin_attack(self, origin: torch.tensor):
        self.original = origin
        self.grad_record = []

    @torch.no_grad()
    def end_attack(self, now: torch.tensor, ksi=16 / 255 / 5):
        '''
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        '''
        fake_grad = (now - self.original)  # x_n-x
        self.outer_momentum = self.mu * self.outer_momentum + fake_grad / torch.norm(fake_grad, p=1)
        now.mul_(0)
        now.add_(self.original)
        now.add_(ksi * self.outer_momentum.sign())
        # patch.add_(ksi * fake_grad)  Without momentum
        now = clamp(now)
        # grad_similarity = cosine_similarity(self.grad_record)
        del self.grad_record
        del self.original
        return now

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


class Adam_CommonWeakness(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10,
                 random_start: bool = False,
                 step_size: float = 1e-3,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu=1,
                 reverse_step_size=16 / 255 / 15,
                 inner_step_size: float = 250,
                 DI=False,
                 TI=False,
                 *args,
                 **kwargs
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        self.reverse_step_size = reverse_step_size
        super(Adam_CommonWeakness, self).__init__(model, *args, **kwargs)
        self.inner_step_size = inner_step_size
        self.DI = DI
        self.TI = TI
        if DI:
            self.aug_policy = transforms.Compose([
                transforms.RandomCrop((int(224 * 0.9), int(224 * 0.9)), padding=224 - int(224 * 0.9)),
            ])
        else:
            self.aug_policy = nn.Identity()
        if TI:
            self.ti = self.gkern().to(self.device)
            self.ti.requires_grad_(False)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        inner_momentum = torch.zeros_like(x)
        self.outer_optimizer = torch.optim.Adam([x], lr=self.step_size, maximize=True)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # first step
            self.begin_attack(x.clone().detach())
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad
            if self.TI:
                grad = self.ti(grad)
            x.requires_grad = False
            if self.targerted_attack:
                x += self.reverse_step_size * grad.sign()
            else:
                x -= self.reverse_step_size * grad.sign()
                # x -= self.reverse_step_size * grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1)
            x = inplace_clamp(x)
            x = inplace_clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            self.outer_optimizer.zero_grad()
            # --------------------------------------------------------------------------------#
            # second step
            x.grad = None
            # self.begin_attack(x.clone().detach())
            for model in self.models:
                x.requires_grad = True
                aug_x = self.aug_policy(x)
                loss = self.criterion(model(aug_x.to(model.device)), y.to(model.device))
                loss.backward()
                grad = x.grad
                self.grad_record.append(grad)
                x.requires_grad = False
                # update
                if self.TI:
                    grad = self.ti(grad)
                if self.targerted_attack:
                    inner_momentum = self.mu * inner_momentum - grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                        N, 1, 1, 1)
                    x += self.inner_step_size * inner_momentum
                else:
                    inner_momentum = self.mu * inner_momentum + grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                        N, 1, 1, 1)
                    x += self.inner_step_size * inner_momentum
                self.outer_optimizer.zero_grad()
                x = inplace_clamp(x)
                x = inplace_clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            x = self.end_attack(x)
            x = inplace_clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x

    @torch.no_grad()
    def begin_attack(self, origin: torch.tensor):
        self.original = origin
        self.grad_record = []

    @torch.no_grad()
    def end_attack(self, now: torch.tensor):
        '''
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        '''
        patch = now
        fake_grad = (patch - self.original)
        patch.mul_(0)
        patch.add_(self.original)
        patch.grad = fake_grad
        self.outer_optimizer.step()
        self.outer_optimizer.zero_grad()
        # patch.add_(ksi * fake_grad)
        patch = inplace_clamp(patch)
        del self.grad_record
        del self.original
        return patch

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
