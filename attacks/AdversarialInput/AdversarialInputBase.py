import torch
from abc import abstractmethod
from typing import List
from torch import Tensor
from math import ceil


class AdversarialInputAttacker():
    def __init__(self, model: List[torch.nn.Module],
                 epsilon=16 / 255,
                 norm='Linf'):
        assert norm in ['Linf', 'L2']
        self.norm = norm
        self.epsilon = epsilon
        self.models = model
        self.init()
        self.model_distribute()
        self.device = torch.device('cuda')
        self.n = len(self.models)

    @abstractmethod
    def attack(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)

    def model_distribute(self):
        '''
        make each model on one gpu
        :return:
        '''
        num_gpus = torch.cuda.device_count()
        models_each_gpu = ceil(len(self.models) / num_gpus)
        for i, model in enumerate(self.models):
            model.to(torch.device(f'cuda:{num_gpus - 1 - i // models_each_gpu}'))
            model.device = torch.device(f'cuda:{num_gpus - 1 - i // models_each_gpu}')

    def init(self):
        # set the model parameters requires_grad is False
        for model in self.models:
            model.requires_grad_(False)
            model.eval()

    def to(self, device: torch.device):
        for model in self.models:
            model.to(device)
            model.device = device
        self.device = device

    def clamp(self, x: Tensor, ori_x: Tensor) -> Tensor:
        B = x.shape[0]
        if self.norm == 'Linf':
            x = torch.clamp(x, min=ori_x - self.epsilon, max=ori_x + self.epsilon)
        elif self.norm == 'L2':
            difference = x - ori_x
            distance = torch.norm(difference.view(B, -1), p=2, dim=1)
            mask = distance > self.epsilon
            if torch.sum(mask) > 0:
                difference[mask] = difference[mask] / distance[mask].view(torch.sum(mask), 1, 1, 1) * self.epsilon
                x = ori_x + difference
        x = torch.clamp(x, min=0, max=1)
        return x
