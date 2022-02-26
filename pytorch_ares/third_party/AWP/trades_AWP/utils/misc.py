#############################################################################################
#  Copyed and modified from https://github.com/bearpaw/pytorch-classification/blob/master/utils/misc.py  #
#############################################################################################

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''

import errno
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


class RunningMeanStd(object):
    def __init__(self, dim=1):
        self._mean = np.zeros(dim)
        self._count = 0
        self._M = np.zeros(dim)
        self.dim = dim

    def update(self, x):
        """
        :param x: [n, d]
        :return:  None
        """
        if isinstance(x, list):
            x = np.array(x)

        avg_a = self._mean
        avg_b = np.mean(x, axis=0)

        count_a = self._count
        count_b = x.shape[0]

        delta = avg_b - avg_a
        m_a = self._M
        m_b = np.var(x, axis=0) * count_b
        M2 = m_a + m_b + np.power(delta, 2) * count_a * count_b / (count_a + count_b)

        self._mean += delta * count_b / (count_a + count_b)
        self._M = M2
        self._count += count_b

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        if self._count == 1:
            return np.ones(self.dim)
        return np.sqrt(self._M / (self._count - 1))


def get_mean_and_std_modified(dataset):
    # Compute the mean and std value online
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    rms = RunningMeanStd(dim=3)

    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        inputs = inputs.detach().cpu().numpy()
        inputs = inputs.transpose((0, 2, 3, 1)).reshape(-1, 3)
        rms.update(inputs)
    return rms.mean, rms.std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count