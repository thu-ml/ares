import torch
import torch.nn as nn
import torch.nn.functional as F

class HELoss(nn.Module):
    def __init__(self, s=None):
        super(HELoss, self).__init__()
        self.s = s

    def forward(self, logits, labels, cm=0):
        numerator = self.s * (torch.diagonal(logits.transpose(0, 1)[labels]) - cm)
        item = torch.cat([torch.cat((logits[i, :y], logits[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * item), dim=1)
        Loss = -torch.mean(numerator - torch.log(denominator))
        return Loss
