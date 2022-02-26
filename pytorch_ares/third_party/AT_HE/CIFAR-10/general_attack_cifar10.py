from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


parser = argparse.ArgumentParser(description='General Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--loss', default='pgd', type=str, help='save frequency')
parser.add_argument('--model-path',
                    default='./checkpoints/pgd.pt',
                    help='model for general attack evaluation')

args = parser.parse_args()
corruptes = ['brightness', 'elastic_transform', 'gaussian_blur', 'impulse_noise',
        'motion_blur', 'shot_noise', 'speckle_noise', 'contrast', 'fog', 'gaussian_noise',
        'jpeg_compression', 'pixelate', 'snow', 'zoom_blur', 'defocus_blur', 'frost', 'glass_blur',
        'saturate', 'spatter']

class CIFAR10_C(Dataset):
    def __init__(self, root, name, transform=None, target_transform=None):
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform
        assert name in corruptes
        file_path = os.path.join(root, 'CIFAR-10-C', name+'.npy')
        lable_path = os.path.join(root, 'CIFAR-10-C', 'labels.npy')
        self.data = np.load(file_path)
        self.targets = np.load(lable_path)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def __len__(self):
        return len(self.data)


# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}


def eval_adv_test_general(model, device, name):
    """
    evaluate model by white-box attack
    """
    # set up data loader
    transform_test = transforms.Compose([transforms.ToTensor(),])
    testset = CIFAR10_C(root='../data', name = name, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model.eval()
    general_acc_total = 0
    for idx, (data, target) in tqdm(enumerate(test_loader)):
        data, target = data.to(device), target.long().to(device)

        X, y = Variable(data, requires_grad=True), Variable(target)
        out = model(X)
        acc_natural = (out.data.max(1)[1] == y.data).float().sum()
        general_acc_total += acc_natural
    print('general_acc_total: ', general_acc_total)
    
def main():
    # general attack
    if args.loss == 'trades' or args.loss == 'pgd' or args.loss == 'alp':
        print("normalize False")
        model = WideResNet().to(device)
    else:
        print("normalize True")
        model = WideResNet(use_FNandWN = True).to(device)
    model.load_state_dict(torch.load(args.model_path))
    
    print('====== test general =====')
    for name in corruptes:
        eval_adv_test_general(model, device, name)
    

if __name__ == '__main__':
    main()
