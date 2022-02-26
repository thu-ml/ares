import torch
import torch.nn as nn

class RobustNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, use_tracked_mean=True, use_tracked_range=True, power=0.2):
        nn.BatchNorm2d.__init__(self, num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                track_running_stats=track_running_stats)
        self.power = power
        self.use_tracked_mean = use_tracked_mean
        self.use_tracked_range = use_tracked_range

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.mean(dim=1)
        min = y.min(dim=1)[0]
        max = y.max(dim=1)[0]
        range = torch.sub(max, min)

        # during validation, whether tracked stat to be used or not
        if self.training is not True:
            if self.use_tracked_mean is True:
                y = y - self.running_mean.view(-1, 1)
            else:
                y = y - mu.view(-1, 1)
            if self.use_tracked_range is True:
                y = y / (self.running_var.view(-1, 1)**self.power + self.eps)
            else:
                y = y / (range.view(-1, 1)**self.power + self.eps)

        # during training tracking will be always be used
        elif self.training is True:
            with torch.no_grad():
                self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu
                self.running_var = (1-self.momentum)*self.running_var + self.momentum*range
            y = y - mu.view(-1, 1)
            y = y / (range.view(-1, 1)**self.power + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)