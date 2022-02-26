# -*- coding: utf-8 -*-

import argparse
import os
import time
import math
import json
import torch
from torch.autograd import Variable as V
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from models.wrn import WideResNet
import numpy as np
from load_corrupted_ours_hierarchical import CIFAR10, CIFAR100
from PIL import Image
import torch.nn as nn


# note: nosgdr, schedule, and epochs are highly related settings

parser = argparse.ArgumentParser(description='Trains WideResNet on CIFAR',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Positional arguments
parser.add_argument('data_path', type=str, help='Root for the Cifar dataset.')
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--gold_fraction', '-gf', type=float, default=0, help='What fraction of the data should be trusted?')
parser.add_argument('--corruption_prob', '-cprob', type=float, default=0.3, help='The label corruption probability.')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif', help='Type of corruption ("unif" or "flip").')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--test_bs', type=int, default=128)
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs. Use when SGDR is off.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--nonlinearity', type=str, default='relu', help='Nonlinearity (relu, elu, gelu).')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# i/o
parser.add_argument('--log', type=str, default='./', help='Log folder.')
# random seed
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use_pretrained_model', action='store_true')
args = parser.parse_args()


np.random.seed(args.seed)

cudnn.benchmark = True  # fire on all cylinders

if args.use_pretrained_model:
    args.epochs = 10

import socket
print()
print("This is on machine:", socket.gethostname())
print()
print(args)
print()

# Init logger
if not os.path.isdir(args.log):
    os.makedirs(args.log)
log = open(os.path.join(args.log, args.dataset + '_log.txt'), 'w')
state = {k: v for k, v in args._get_kwargs()}
log.write(json.dumps(state) + '\n')

# Init dataset
if not os.path.isdir(args.data_path):
    os.makedirs(args.data_path)

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data = CIFAR10(
        args.data_path, True, False, 0, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True, shuffle_indices=None)
    train_data_deterministic = CIFAR10(
        args.data_path, True, False, 0, args.corruption_prob, args.corruption_type,
        transform=test_transform, download=True, shuffle_indices=None)
    test_data = CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 10

elif args.dataset == 'cifar100':
    train_data = CIFAR100(
        args.data_path, True, False, 0, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True, shuffle_indices=None)
    train_data_deterministic = CIFAR100(
        args.data_path, True, False, 0, args.corruption_prob, args.corruption_type,
        transform=test_transform, download=True, shuffle_indices=None)
    test_data = CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 100


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, transform):
        # assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data_tensor[index], self.target_tensor[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.target_tensor.size()[0]

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

train_loader_deterministic = torch.utils.data.DataLoader(
    train_data_deterministic,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)


# Init checkpoints
if not os.path.isdir(args.save):
    os.makedirs(args.save)

class FineTuneModel(nn.Module):
        def __init__(self, model):
            super(FineTuneModel, self).__init__()

            self.trunk = nn.Sequential(*list(model.children())[:-1])
            # self.classifier = list(model.children())[-1]

            self.classifier = nn.Linear(128, num_classes, bias=False)
            self.unfreeze(True)

        def unfreeze(self, unfreeze):
            for p in self.trunk.parameters():
                p.requires_grad = unfreeze

        def forward(self, x):
            x = self.trunk(x)
            x = F.avg_pool2d(x, 8)
            x = x.view(x.size(0), -1)
            return self.classifier(x)   #, x

# Init model, criterion, and optimizer
if args.use_pretrained_model:
    net = WideResNet(args.layers, 1000, args.widen_factor, dropRate=0)

    # net = nn.DataParallel(net)

    # Load pretrained model
    net.load_state_dict(torch.load('./snapshots/baseline/imagenet_wrn_baseline_epoch_99.pt'))

    # net = net.module

    net = FineTuneModel(net)
    state['learning_rate'] = 0.01
else:
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

print('Loaded pretrained model for first phase of training')
#

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()

torch.manual_seed(args.seed)
if args.ngpu > 0:
    torch.cuda.manual_seed(args.seed)


optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()),
                            state['learning_rate'], momentum=state['momentum'],
                            weight_decay=state['decay'], nesterov=True)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


# train function (forward, backward, update)
def train(no_correction=True, C_hat_transpose=None, scheduler=scheduler):
    net.train()     # enter train mode
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = V(data.cuda()), V(target.cuda())

        # forward
        output = net(data)

        # backward
        scheduler.step()
        optimizer.zero_grad()
        if no_correction:
            loss = F.cross_entropy(output, target)
        else:
            pre1 = C_hat_transpose[torch.cuda.LongTensor(target.data)]
            pre2 = torch.mul(F.softmax(output), pre1)
            loss = -(torch.log(pre2.sum(1))).mean()
        loss.backward()
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.95 + loss.item() * 0.05

    state['train_loss'] = loss_avg


# test function (forward only)
def test():
    torch.set_grad_enabled(False)
    net.eval()
    loss_avg = 0.0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()

        # forward
        output = net(data)
        loss = F.cross_entropy(output, target)

        # accuracy
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum().item()

        # test loss average
        loss_avg += loss.item()

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)
    torch.set_grad_enabled(True)


# Main loop
for epoch in range(args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()
    train(scheduler=scheduler)
    print('Epoch', epoch, '| Time Spent:', round(time.time() - begin_epoch, 2))

    test()

    log.write('%s\n' % json.dumps(state))
    log.flush()
    print(state)


print('\nNow retraining with correction\n')


def get_C_hat_transpose():
    probs = []
    net.eval()
    for batch_idx, (data, target) in enumerate(train_loader_deterministic):
        data, target = V(data.cuda()),\
                       V(target.cuda())

        # forward
        output = net(data)
        pred = F.softmax(output)
        probs.extend(list(pred.data.cpu().numpy()))

    probs = np.array(probs, dtype=np.float32)
    C_hat = np.zeros((num_classes, num_classes))
    for label in range(num_classes):
        class_probs = probs[:, label]
        if args.dataset == 'cifar10':
            threshold = np.percentile(class_probs, 97, interpolation='higher')
            class_probs[class_probs >= threshold] = 0

        C_hat[label] = probs[np.argmax(class_probs)]

        # C_hat[label] = probs[np.argsort(class_probs)][-1]

    return C_hat.T.astype(np.float32)

C_hat_transpose = torch.from_numpy(get_C_hat_transpose())
C_hat_transpose = V(C_hat_transpose.cuda(), requires_grad=False)


# /////// Resetting the network ////////
state = {k: v for k, v in args._get_kwargs()}


if args.use_pretrained_model:
    # Load pretrained model
    net = WideResNet(args.layers, 1000, args.widen_factor, dropRate=0)
    # net = nn.DataParallel(net)
    net.load_state_dict(torch.load('./snapshots/baseline/imagenet_wrn_baseline_epoch_99.pt'))
    # net = net.module
    net = FineTuneModel(net).cuda()
    state['learning_rate'] = 0.01
else:
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).cuda()

optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()),
                            state['learning_rate'], momentum=state['momentum'],
                            weight_decay=state['decay'], nesterov=True)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))

print('Loaded pretrained model for second phase of training')
#


# Main loop
for epoch in range(args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()
    train(no_correction=False, C_hat_transpose=C_hat_transpose, scheduler=scheduler)
    print('Epoch', epoch, '| Time Spent:', round(time.time() - begin_epoch, 2))

    test()

    log.write('%s\n' % json.dumps(state))
    log.flush()
    print(state)

log.close()

try: os.remove(os.path.join(
    args.save,
    args.dataset+'_'+str(args.gold_fraction) + str(args.corruption_prob) + args.corruption_type + '_init.pytorch'))
except: True
