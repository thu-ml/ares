import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset


class CustomDatasetFromNumpy(Dataset):
    def __init__(self, img, label, transform):
        self.img = img
        self.label = label
        self.transform = transform
        self.len = len(self.img)

    def __getitem__(self, index):
        img_tensor = transforms.ToPILImage()(self.img[index])
        img_tensor = self.transform(img_tensor)
        label_tensor = self.label[index]
        return (img_tensor, label_tensor)

    def __len__(self):
        return self.len


def xe_with_one_hot(out, target):
    """
        out: [N,k] dim tensor with output logits.
        target: [N,k] dim tensor with ground truth probs.
        
        return: calcuate mean(-1*sum(p_i*out_i))
    """
    log_prob = nn.LogSoftmax(dim=1)(out)
    loss = -1 * torch.sum(log_prob * target, dim=1)
    loss = torch.sum(loss) / len(loss)
    return loss

