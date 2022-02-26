import torch
import torch.nn as nn

import numpy as np


def get_lr_policy(lr_schedule):
    """Implement a new schduler directly in this file. 
    Args should contain a single choice for learning rate scheduler."""

    d = {
        "constant": constant_schedule,
        "cosine": cosine_schedule,
        "step": step_schedule,
    }
    return d[lr_schedule]


def get_optimizer(model, args):
    if args.optimizer == "sgd":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    elif args.optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd,)
    elif args.optimizer == "rmsprop":
        optim = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    else:
        print(f"{args.optimizer} is not supported.")
        sys.exit(0)
    return optim


def new_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def constant_schedule(optimizer, args):
    def set_lr(epoch, lr=args.lr, epochs=args.epochs):
        if epoch < args.warmup_epochs:
            lr = args.warmup_lr

        new_lr(optimizer, lr)

    return set_lr


def cosine_schedule(optimizer, args):
    def set_lr(epoch, lr=args.lr, epochs=args.epochs):
        if epoch < args.warmup_epochs:
            a = args.warmup_lr
        else:
            epoch = epoch - args.warmup_epochs
            a = lr * 0.5 * (1 + np.cos((epoch - 1) / epochs * np.pi))

        new_lr(optimizer, a)

    return set_lr


def step_schedule(optimizer, args):
    def set_lr(epoch, lr=args.lr, epochs=args.epochs):
        if epoch < args.warmup_epochs:
            a = args.warmup_lr
        else:
            epoch = epoch - args.warmup_epochs

        a = lr
        if epoch >= 0.75 * epochs:
            a = lr * 0.1
        if epoch >= 0.9 * epochs:
            a = lr * 0.01
        if epoch >= epochs:
            a = lr * 0.001

        new_lr(optimizer, a)

    return set_lr
