import torch
import collections.abc as container_abcs


def check_imgs(adv, x, norm):
    delta = (adv - x).view(adv.shape[0], -1)
    if norm == 'Linf':
        res = delta.abs().max(dim=1)[0]
    elif norm == 'L2':
        res = (delta ** 2).sum(dim=1).sqrt()
    elif norm == 'L1':
        res = delta.abs().sum(dim=1)

    str_det = 'max {} pert: {:.5f}, nan in imgs: {}, max in imgs: {:.5f}, min in imgs: {:.5f}'.format(
        norm, res.max(), (adv != adv).sum(), adv.max(), adv.min())
    print(str_det)

    return str_det

def L1_norm(x, keepdim=False):
    z = x.abs().view(x.shape[0], -1).sum(-1)
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L2_norm(x, keepdim=False):
    z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L0_norm(x):
    return (x != 0.).view(x.shape[0], -1).sum(-1)

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)
