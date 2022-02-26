'''

Improving Adversarial Robustness via Probabilistically Compact Loss with Logit Constraints (AAAI2021)
Paper link: https://arxiv.org/abs/2012.07688

Modified by Xiaofeng Mao 
2021.7.22
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class HardProbCompactLoss(nn.Module):
    """Margin Loss
    Args:
        num_classes (int): number of classes.
    """
    def __init__(self, num_classes=10, margin=0.8, use_gpu=True):
        super(HardProbCompactLoss, self).__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.use_gpu = use_gpu


    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        x = F.softmax(x,dim=1)
        p_gt = torch.mul(mask.float(), x).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
        
        diff = self.margin + x - p_gt
        zeros = torch.zeros(batch_size, self.num_classes).cuda()
        loss = (torch.max(zeros, diff).sum() / batch_size - self.margin)  / (self.num_classes - 1)

    
        return loss

class SoftProbCompactLoss(nn.Module):
    """Margin Loss
    Args:
        num_classes (int): number of classes.
    """
    def __init__(self, num_classes=10, use_gpu=True):
        super(SoftProbCompactLoss, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu


    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        x_true = torch.mul(mask.float(), x)
        p_gt = x_true.sum(dim=1, keepdim=True)#.expand(batch_size, self.num_classes)

        gt_mask = -1e3 * mask

        # x exclude true label
        x_false = x - x_true + gt_mask # replace true label logit with -1e3
        # use softmax to pick the maximum false prob
        x_false_max = F.softmax(x_false, dim=1)

        diff = x_false_max * (p_gt - x)
        zeros = torch.zeros(batch_size, self.num_classes).cuda()
        loss = torch.max(zeros, diff).sum() / batch_size
    
        return loss