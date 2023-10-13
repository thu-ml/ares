"""
reference:
https://github.com/yuyang-long/SSA
"""

import os
from torch.autograd import Variable as V
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image

import numpy as np
import torch


def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.fft.fft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1).real.view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape).real


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def save_image(images, names, output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    for i, name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)


from .AdversarialInputBase import AdversarialInputAttacker
from attacks.utils import *
from torch import nn
from typing import Callable, List
from tqdm import tqdm


class SpectrumSimulationAttack(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
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
        super(SpectrumSimulationAttack, self).__init__(model, *args, **kwargs)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        """
        The attack algorithm of our proposed Spectrum Simulate Attack
        :param images: the input images
        :param gt: ground-truth
        :param model: substitute model
        :param mix: the mix the clip operation
        :param max: the max the clip operation
        :return: the adversarial images
        """
        ori_x = x.clone()
        momentum = self.mu
        num_iter = self.total_step
        eps = self.epsilon
        alpha = self.step_size
        grad = 0
        rho = 0.5
        N = 20
        sigma = 16

        for i in tqdm(range(num_iter)):
            noise = 0
            for n in range(N):
                x.requires_grad = True
                gauss = torch.randn(*x.shape) * (sigma / 255)
                gauss = gauss.cuda()
                x_dct = dct_2d(x + gauss).cuda()
                mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
                x_idct = idct_2d(x_dct * mask)
                x_idct = V(x_idct, requires_grad=True)
                logit = 0
                for model in self.models:
                    logit += model(x_idct.to(model.device)).to(x_idct.device)
                loss = self.criterion(logit, y)
                loss.backward()
                x.requires_grad = False
                noise += x_idct.grad.data
                x.grad = None
            noise = noise / N
            noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
            noise = momentum * grad + noise
            grad = noise

            x = x + alpha * torch.sign(noise)
            x = self.clamp(x, ori_x)
        return x


class SSA_CommonWeakness(AdversarialInputAttacker):
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
        self.reverse_step_size = reverse_step_size
        super(SSA_CommonWeakness, self).__init__(model, *args, **kwargs)
        self.inner_step_size = inner_step_size

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

        for _ in tqdm(range(self.total_step)):
            # # --------------------------------------------------------------------------------#
            # # first step
            # self.begin_attack(x.clone().detach())
            # x.requires_grad = True
            # logit = 0
            # for model in self.models:
            #     logit += model(x.to(model.device)).to(x.device)
            # loss = self.criterion(logit, y)
            # loss.backward()
            # grad = x.grad
            # x.requires_grad = False
            # if self.targerted_attack:
            #     x += self.reverse_step_size * grad.sign()
            # else:
            #     x -= self.reverse_step_size * grad.sign()
            # x = self.clamp(x, original_x)
            # # --------------------------------------------------------------------------------#
            # # second step
            x.grad = None
            self.begin_attack(x.clone().detach())
            for model in self.models:
                x.requires_grad = True
                grad = self.get_grad(x, y, model)
                self.grad_record.append(grad)
                x.requires_grad = False
                # update
                if self.targerted_attack:
                    inner_momentum = self.mu * inner_momentum - grad / (
                                torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                                    N, 1, 1, 1) + 1e-5)
                    x += self.inner_step_size * inner_momentum
                else:
                    inner_momentum = self.mu * inner_momentum + grad / (
                                torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                                    N, 1, 1, 1) + 1e-5)
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
        del self.grad_record
        del self.original
        return patch

    def get_grad(self, x, y, model):
        rho = 0.5
        N = 20
        sigma = 16
        noise = 0
        for n in range(N):
            x.requires_grad = True
            gauss = torch.randn(*x.shape) * (sigma / 255)
            gauss = gauss.cuda()
            x_dct = dct_2d(x + gauss).cuda()
            mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
            x_idct = idct_2d(x_dct * mask)
            x_idct = V(x_idct, requires_grad=True)
            logit = model(x_idct.to(model.device)).to(x_idct.device)
            loss = self.criterion(logit, y)
            loss.backward()
            x.requires_grad = False
            noise += x_idct.grad.data
            x.grad = None
        noise = noise / N
        return noise
