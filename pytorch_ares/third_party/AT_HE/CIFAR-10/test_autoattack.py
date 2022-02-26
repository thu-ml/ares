'''
This script depends on https://github.com/fra31/auto-attack
'''
import torch
import torch.nn as nn

import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from autoattack import AutoAttack
from models.wideresnet import *
import argparse

parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--name', type=str, default='pgd')
parser.add_argument('--norm', type=str, default='Linf')
parser.add_argument('--epsilon', type=float, default=0.031372549)
parser.add_argument('--model', type=str, default='WideResNet')
parser.add_argument('--model-path', type=str, default='')
parser.add_argument('--use_FNandWN', action='store_true')
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--n-ex', type=int, default=10000)
parser.add_argument('--widen_factor', type=int, default=20, choices=[10, 20])
parser.add_argument('--logfile', default='result.log', type=str, help='save log')                  

args = parser.parse_args()

use_cuda =  torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
l = [x for (x, y) in test_loader]
x_test = torch.cat(l, 0)
l = [y for (x, y) in test_loader]
y_test = torch.cat(l, 0)

# set models
if args.model == 'WideResNet':
    if args.use_FNandWN:
        model = WideResNet(34, 10, widen_factor=args.widen_factor, dropRate=0.0, use_FNandWN=True, i_normalize=True)
    else:
        model = WideResNet(34, 10, widen_factor=args.widen_factor, dropRate=0.0, i_normalize=True)
    checkpoint = torch.load(args.model_path)
    # load checkpoint
    print('test_acc:{}, test_robust_acc:{}'.format(checkpoint['test_acc'], checkpoint['test_robust_acc']))
    state_dict = checkpoint['state_dict']
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(state_dict)
    model.eval()
else:
    raise ValueError("Unknown model")
adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.logfile)

with torch.no_grad():
    x_adv = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex], bs=args.batch_size)
    