import torch
from attacks.utils import *
from torch import nn
from typing import Callable, List
from .AdversarialInputBase import AdversarialInputAttacker
from torchvision import transforms
from .utils import *
import numpy as np
from scipy import stats as st


class VMI_FGSM(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 10,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 *args, **kwargs
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(VMI_FGSM, self).__init__(model, *args, **kwargs)

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
            # x.requires_grad = True
            # logit = 0
            # for model in self.models:
            #     logit += model(x.to(model.device)).to(x.device)
            # loss = self.criterion(logit, y)
            # loss.backward()
            # grad = x.grad
            # x.requires_grad = False
            # update
            grad = self.calculate_v(x, y)
            if self.targerted_attack:
                momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            x = clamp(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x

    def calculate_v(self, x: torch.tensor, y: torch.tensor, N=20, beta=1.5):
        """

        :param x:  B, C, H, D
        :param y:
        :param N:
        :param beta:
        :return:
        """
        B, C, H, D = x.shape
        x = x.reshape(1, B, C, H, D)
        x = x.repeat(N, 1, 1, 1, 1)  # N, B, C, H, D
        ranges = beta * self.epsilon
        now = x + (torch.rand_like(x) - 0.5) * 2 * ranges
        now = now.view(N * B, C, H, D)
        now.requires_grad = True
        logit = 0
        for model in self.models:
            logit += model(now.to(model.device)).to(now.device)
        loss = self.criterion(logit, y.repeat(N))
        loss.backward()
        v = now.grad.view(N, B, C, H, D)  # N, B, C, H, D
        v = v.mean(0)
        return v


class VMI_Inner_CommonWeakness(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10,
                 random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu=1,
                 outer_optimizer=None,
                 reverse_step_size=16 / 255 / 15,
                 inner_step_size: float = 250,
                 DI=False,
                 TI=False,
                 *args, **kwargs
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        self.outer_optimizer = outer_optimizer
        self.reverse_step_size = reverse_step_size
        super(VMI_Inner_CommonWeakness, self).__init__(model, *args, **kwargs)
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
            # first step
            self.begin_attack(x.clone().detach())
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            if self.targerted_attack:
                pass
            else:
                x -= self.reverse_step_size * grad.sign()
            #     # x -= self.reverse_step_size * grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1)
            # # --------------------------------------------------------------------------------#
            # # second step
            x.grad = None
            # self.begin_attack(x.clone().detach())
            for model in self.models:
                grad = self.calculate_v(x, y, model)
                self.grad_record.append(grad)
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

    def calculate_v(self, x: torch.tensor, y: torch.tensor, model: nn.Module, N=8, beta=1.5):
        B, C, H, D = x.shape
        x = x.reshape(1, B, C, H, D)
        x = x.repeat(N, 1, 1, 1, 1)
        ranges = beta * self.epsilon
        now = x + (torch.rand_like(x) - 0.5) * 2 * ranges
        now = now.view(N * B, C, H, D)
        now.requires_grad = True
        logit = model(now.to(model.device)).to(now.device)
        loss = self.criterion(logit, y.repeat(N))
        loss.backward()
        v = now.grad.view(N, B, C, H, D)
        v = v.mean(0)
        return v

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


class VMI_Outer_CommonWeakness(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10,
                 random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu=1,
                 outer_optimizer=None,
                 reverse_step_size=16 / 255 / 15,
                 inner_step_size: float = 250,
                 DI=False,
                 TI=False,
                 *args, **kwargs
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        self.outer_optimizer = outer_optimizer
        self.reverse_step_size = reverse_step_size
        super(VMI_Outer_CommonWeakness, self).__init__(model, *args, **kwargs)
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
        original_x = x.clone()
        inner_momentum = None
        self.outer_momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # first step
            non_perturbed_x = x.clone()
            x, y = self.get_input(x, y)
            if inner_momentum is None:
                inner_momentum = torch.zeros_like(x)
                NB = x.shape[0]
            self.begin_attack(x.clone().detach())
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            if self.targerted_attack:
                pass
            else:
                x -= self.reverse_step_size * grad.sign()
                # x -= self.reverse_step_size * grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1)
            x = clamp(x)
            x = clamp(x, original_x.repeat(8, 1, 1, 1) - self.epsilon, original_x.repeat(8, 1, 1, 1) + self.epsilon)

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
                    inner_momentum = self.mu * inner_momentum - grad / torch.norm(grad.reshape(NB, -1), p=2,
                                                                                  dim=1).view(
                        NB, 1, 1, 1)
                    x += self.inner_step_size * inner_momentum
                else:
                    inner_momentum = self.mu * inner_momentum + grad / torch.norm(grad.reshape(NB, -1), p=2,
                                                                                  dim=1).view(
                        NB, 1, 1, 1)
                    x += self.inner_step_size * inner_momentum
                x = clamp(x)
                x = clamp(x, original_x.repeat(8, 1, 1, 1) - self.epsilon, original_x.repeat(8, 1, 1, 1) + self.epsilon)
            x = self.end_attack(x, non_perturbed_x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x

    @torch.no_grad()
    def begin_attack(self, origin: torch.tensor):
        self.original = origin
        self.grad_record = []

    @torch.no_grad()
    def end_attack(self, now: torch.tensor, non_perturbed_x: torch.tensor, ksi=16 / 255 / 5):
        '''
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        '''
        B, C, H, D = non_perturbed_x.shape
        patch = now
        if self.outer_optimizer is None:
            fake_grad = (patch - self.original)  # B*N, C, H, D
            fake_grad = fake_grad.reshape(-1, B, C, H, D)
            fake_grad = fake_grad.mean(0)
            self.outer_momentum = self.mu * self.outer_momentum + fake_grad / torch.norm(fake_grad, p=1)
            non_perturbed_x.add_(ksi * self.outer_momentum.sign())
            # patch.add_(ksi * fake_grad)
        non_perturbed_x = clamp(non_perturbed_x)
        # grad_similarity = cosine_similarity(self.grad_record)
        del self.grad_record
        del self.original
        return non_perturbed_x

    def get_input(self, x: torch.tensor, y: torch.tensor, N=8, beta=1):
        B, C, H, D = x.shape
        x = x.reshape(1, B, C, H, D)
        x = x.repeat(N, 1, 1, 1, 1)
        ranges = beta * self.epsilon
        now = x + (torch.rand_like(x) - 0.5) * 2 * ranges
        # now = x + torch.randn_like(x) * ranges
        now = now.view(N * B, C, H, D)
        if now.shape[0] != y.shape[0]:
            y = y.repeat(N)
        return now, y

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
