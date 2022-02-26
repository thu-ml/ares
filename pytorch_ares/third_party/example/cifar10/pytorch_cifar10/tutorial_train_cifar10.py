# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
# from ....attack_torch import FGSM
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../'))
from attack_torch import FGSM
import argparse
from attack_benchmark.utils import TRAINED_MODEL_PATH

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--mode', default="cln", help="cln | adv")
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--log_interval', default= 200, type=int)
args = parser.parse_args()

device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=os.path.join(os.path.dirname(__file__),'../../../../data/CIFAR10'), train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=10, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=os.path.join(os.path.dirname(__file__),'../../../../data/CIFAR10'), train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=10, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

if args.mode == "cln":
        flag_advtrain = False
        nb_epoch = 200
        model_filename = "simpledla_ckpt.pt"
elif args.mode == "adv":
    flag_advtrain = True
    nb_epoch = 200
    model_filename = "simpledlaadv_ckpt.pt"
else:
    raise
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= nb_epoch)


if flag_advtrain:
    
    adversary = FGSM(net=net, criterion=criterion, targeted=False, eps=0.03, alpha=1, iteration=3,x_val_min=-1, x_val_max=1)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_clnloss = 0
    clncorrect = 0
    if flag_advtrain:
        test_advloss = 0
        advcorrect = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            output = net(inputs)
        test_clnloss += criterion(output, targets).item()
        pred = output.max(1, keepdim=True)[1]
        clncorrect += pred.eq(targets.view_as(pred)).sum().item()
        if flag_advtrain:
            advdata, _, _ = adversary.forward(inputs, targets)
            with torch.no_grad():
                outputs = net(advdata)
            test_advloss += criterion(outputs, targets).item()
            pred = outputs.max(1, keepdim=True)[1]
            advcorrect +=pred.eq(targets.view_as(pred)).sum().item()
    test_clnloss /= len(testloader.dataset)
    clean_acc = 100.* clncorrect / len(testloader.dataset)
    print('\nTest set: avg clnean loss: {:.4f},'
          'clnean acc: {}/{} ({:.0f}%)\n'.format(
              test_clnloss, clncorrect, len(testloader.dataset),
              clean_acc))
    if flag_advtrain:
        test_advloss /= len(testloader.dataset)
        print('\nTest set: avg adv loss: {:.4f},'
          'adv acc: {}/{} ({:.0f}%)\n'.format(
              test_advloss, advcorrect, len(testloader.dataset),
              100.* advcorrect / len(testloader.dataset)))
    if args.mode == "cln":
        if clean_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': clean_acc,
                'epoch': epoch,
            }
            if not os.path.isdir(os.path.join(os.path.dirname(__file__),'../../../../attack_benchmark/checkpoint')):
                os.mkdir('checkpoint')
            torch.save(state, os.path.join(os.path.dirname(__file__),'../../../../attack_benchmark/checkpoint/simpledla_ckpt.pth'))
            best_acc = clean_acc
    


for epoch in range(start_epoch, nb_epoch):
    train(epoch)
    test(epoch)
    scheduler.step()
    torch.save(net.state_dict(), os.path.join(TRAINED_MODEL_PATH, model_filename))





# for epoch in range(nb_epoch):
#     net.train()
#     for batch_idx, (data, target) in enumerate(trainloader):
#         data, target = data.to(device), target.to(device)
#         ori = data
#         if flag_advtrain:
#             # when performing attack, the model needs to be in eval mode
#             # also the parameters should NOT be accumulating gradients
#             with ctx_noparamgrad_and_eval(net):
#                 data = adversary.fgsm(data, target)

#         optimizer.zero_grad()
#         output = net(data)
#         loss = F.cross_entropy(
#             output, target, reduction='elementwise_mean')
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx *
#                 len(data), len(trainloader.dataset),
#                 100. * batch_idx / len(trainloader), loss.item()))

#     net.eval()
#     test_clnloss = 0
#     clncorrect = 0

#     if flag_advtrain:
#         test_advloss = 0
#         advcorrect = 0

#     for clndata, target in testloader:
#         clndata, target = clndata.to(device), target.to(device)
#         with torch.no_grad():
#             output = net(clndata)
#         test_clnloss += F.cross_entropy(
#             output, target, reduction='sum').item()
#         pred = output.max(1, keepdim=True)[1]
#         clncorrect += pred.eq(target.view_as(pred)).sum().item()

#         if flag_advtrain:
#             advdata = adversary.fgsm(clndata, target)
#             with torch.no_grad():
#                 output = net(advdata)
#             test_advloss += F.cross_entropy(
#                 output, target, reduction='sum').item()
#             pred = output.max(1, keepdim=True)[1]
#             advcorrect += pred.eq(target.view_as(pred)).sum().item()

#     test_clnloss /= len(testloader.dataset)
#     print('\nTest set: avg cln loss: {:.4f},'
#             ' cln acc: {}/{} ({:.0f}%)\n'.format(
#                 test_clnloss, clncorrect, len(testloader.dataset),
#                 100. * clncorrect / len(testloader.dataset)))
#     if flag_advtrain:
#         test_advloss /= len(testloader.dataset)
#         print('Test set: avg adv loss: {:.4f},'
#                 ' adv acc: {}/{} ({:.0f}%)\n'.format(
#                     test_advloss, advcorrect, len(testloader.dataset),
#                     100. * advcorrect / len(testloader.dataset)))

# torch.save(
#     net.state_dict(),
#     os.path.join(TRAINED_MODEL_PATH, model_filename))
