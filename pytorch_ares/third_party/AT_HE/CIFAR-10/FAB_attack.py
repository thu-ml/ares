# Please setup this dependent package 'https://github.com/BorealisAI/advertorch'
import torch.nn as nn

from advertorch.attacks import LinfPGDAttack, FABAttack, LinfFABAttack
from advertorch.attacks.utils import multiple_mini_batch_attack
from advertorch_examples.utils import get_cifar10_test_loader

from models.wideresnet import *
import argparse

parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--model-path', type=str, help='model path')
parser.add_argument('--loss', type=str, default='pgdHE')
parser.add_argument('--attack', type=str, default='FAB')
parser.add_argument('--iters', type=int, default=20)
parser.add_argument('--norm', type=str, default='Linf')

args = parser.parse_args()

eps=8./255.
if args.loss == 'trades' or args.loss == 'pgd' or args.loss == 'alp':
    print("normalize False")
    model = WideResNet()
else:
    print("normalize True")
    model = WideResNet(use_FNandWN = True)

model.load_state_dict(torch.load(args.model_path))
loader = get_cifar10_test_loader(batch_size=100)

if args.attack == 'FAB':
    adversary = LinfFABAttack(model, n_restarts=1, n_iter=args.iters, 
        alpha_max=0.1, eta=1.05, beta=0.9, loss_fn=None, verbose=False)
else: 
    adversary = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
        nb_iter=20, eps_iter=2. / 255, rand_init=False, clip_min=0.0, clip_max=1.0,
        targeted=False)

label, pred, advpred, dist = multiple_mini_batch_attack(
    adversary, loader, device="cuda",norm='Linf')

print("Natural Acc: {:.2f}, Robust Acc: {:.2f}".format(100. * (label == pred).sum().item() / len(label),
    100. * (dist > eps).sum().item()/len(dist)))
