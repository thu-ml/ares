### PGD implementation

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
import logging


def pgd(model,
        X,
        y,
        epsilon=8 / 255,
        num_steps=20,
        step_size=0.01,
        random_start=True):
    out = model(X)
    is_correct_natural = (out.max(1)[1] == y).float().cpu().numpy()
    perturbation = torch.zeros_like(X, requires_grad=True)

    if random_start:
        perturbation = torch.rand_like(X, requires_grad=True)
        perturbation.data = perturbation.data * 2 * epsilon - epsilon

    is_correct_adv = []
    opt = optim.SGD([perturbation], lr=1e-3)  # This is just to clear the grad

    for _ in range(num_steps):
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X + perturbation), y)

        loss.backward()

        perturbation.data = (
            perturbation + step_size * perturbation.grad.detach().sign()).clamp(
            -epsilon, epsilon)
        perturbation.data = torch.min(torch.max(perturbation.detach(), -X),
                                      1 - X)  # clip X+delta to [0,1]
        X_pgd = Variable(torch.clamp(X.data + perturbation.data, 0, 1.0),
                         requires_grad=False)
        is_correct_adv.append(np.reshape(
            (model(X_pgd).max(1)[1] == y).float().cpu().numpy(),
            [-1, 1]))

    is_correct_adv = np.concatenate(is_correct_adv, axis=1)
    return is_correct_natural, is_correct_adv


