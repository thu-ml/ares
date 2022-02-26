import torch
from torch import nn
import torch

def loss_adv(loss_name, outputs, labels, target_labels, target, device):
    if loss_name=="ce":
        loss = nn.CrossEntropyLoss()
        
        if target:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

    elif loss_name =='cw':
        if target:
            one_hot_labels = torch.eye(len(outputs[0]))[target_labels].to(device)
            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            cost = -torch.clamp((i-j), min=0)  # -self.kappa=0
            cost = cost.sum()
        else:
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)
            i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            cost = -torch.clamp((j-i), min=0)  # -self.kappa=0
            cost = cost.sum()
    return cost



class Normalize(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:,i] = (x[:,i] - self.mean[i])/self.std[i]

        return x



def margin_loss(outputs, labels, target_labels, targeted, device):
    if targeted:
        one_hot_labels = torch.eye(len(outputs[0]))[target_labels].to(device)
        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        cost = -torch.clamp((i-j), min=0)  # -self.kappa=0
    else:
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)
        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        cost = -torch.clamp((j-i), min=0)  # -self.kappa=0
    return cost.sum()