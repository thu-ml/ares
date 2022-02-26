#############################################################################################
#  Copyed from https://github.com/bearpaw/pytorch-classification/blob/master/utils/eval.py  #
#############################################################################################

from __future__ import print_function, absolute_import

__all__ = ['accuracy']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # output: [B, d]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # pred: [B, len(topk)] -> [len(topk), B]
    # target.view(1, -1).expand_as(pred): [len(topk), B]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # correct: [len(topk), B]

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    import torch
    outputs = torch.tensor([[0.3, 0.4], [1.0, 0.8], [0.9, 0.6]])
    targets = torch.tensor([1, 0, 1])
    print(accuracy(outputs, targets, topk=(1, 2)))