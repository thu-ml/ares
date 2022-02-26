# This module is adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import init_paths
import argparse
import os
import time
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import math
import numpy as np
from utils import *
from validation import validate, validate_pgd
import models
import re



def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
    parser.add_argument('--FN', default=False, type=bool,
                    help='whether use FN')
    parser.add_argument('--WN', default=False, type=bool,
                    help='whether use WN')
    parser.add_argument('--s_HE', default=10.0, type=float,
                    help='the vaule of s in HE')
    parser.add_argument('--angular_margin_HE', default=0.2, type=float,
                    help='the vaule of angular margin in HE')
    parser.add_argument('--output_prefix', default='free_adv', type=str,
                    help='prefix used to define output path')
    parser.add_argument('-c', '--config', default='configs.yml', type=str, metavar='Path',
                    help='path to the config file (default: configs.yml)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    parser.add_argument('--print_gradients', default=False, type=bool,
                    help='whether calculate intermediate gradients')
    return parser.parse_args()



# Parase config file and initiate logging
configs = parse_config_file(parse_args())
logger = initiate_logger(configs.output_name)

cudnn.benchmark = True

angular_margin_HE = configs.angular_margin_HE
s_HE = configs.s_HE
print('maring: ', angular_margin_HE)
print('s: ', s_HE)


# calculate the gradients w.r.t conv and bn layers
if configs.print_gradients == True:
    # save name and parameters of bn and conv
    name_all = []
    para_all = {}

    # save mean and std of gradient norms for each epoch, seperately for clean and adv data
    mean_per_epoch_cle = {}
    std_per_epoch_cle = {}

    mean_per_epoch_adv = {}
    std_per_epoch_adv = {}




def main():
    global name_all
    global para_all
    global mean_per_epoch_cle
    global std_per_epoch_cle
    global mean_per_epoch_adv
    global std_per_epoch_adv
    # Scale and initialize the parameters
    best_prec1 = 0
    configs.TRAIN.epochs = int(math.ceil(configs.TRAIN.epochs / configs.ADV.n_repeats))
    configs.ADV.fgsm_step /= configs.DATA.max_color_value
    configs.ADV.clip_eps /= configs.DATA.max_color_value
    
    # Create output folder
    if not os.path.isdir(os.path.join('trained_models', configs.output_name)):
        os.makedirs(os.path.join('trained_models', configs.output_name))
    
    # Log the config details
    logger.info(pad_str(' ARGUMENTS '))
    for k, v in configs.items(): print('{}: {}'.format(k, v))
    logger.info(pad_str(''))

    
    # Create the model
    if configs.pretrained:
        print("=> using pre-trained model '{}'".format(configs.TRAIN.arch))
        model = models.__dict__[configs.TRAIN.arch](pretrained=True, useFN=configs.FN, useWN=configs.WN)
    else:
        print("=> creating model '{}'".format(configs.TRAIN.arch))
        model = models.__dict__[configs.TRAIN.arch](useFN=configs.FN, useWN=configs.WN)

    # Wrap the model into DataParallel
    model = torch.nn.DataParallel(model).cuda()
    
    # Criterion:
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Optimizer:
    optimizer = torch.optim.SGD(model.parameters(), configs.TRAIN.lr,
                                momentum=configs.TRAIN.momentum,
                                weight_decay=configs.TRAIN.weight_decay)
    
    # Resume if a valid checkpoint path is provided
    if configs.resume:
        if os.path.isfile(configs.resume):
            print("=> loading checkpoint '{}'".format(configs.resume))
            checkpoint = torch.load(configs.resume)
            configs.TRAIN.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(configs.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(configs.resume))

            
    # Initiate data loaders
    traindir = os.path.join(configs.data, 'train')
    valdir = os.path.join(configs.data, 'val')
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(configs.DATA.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=configs.DATA.batch_size, shuffle=True,
        num_workers=configs.DATA.workers, pin_memory=True, sampler=None)
    
    normalize = transforms.Normalize(mean=configs.TRAIN.mean,
                                    std=configs.TRAIN.std)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(configs.DATA.img_size),
            transforms.CenterCrop(configs.DATA.crop_size),
            transforms.ToTensor(),
        ])),
        batch_size=configs.DATA.batch_size, shuffle=False,
        num_workers=configs.DATA.workers, pin_memory=True)

    # If in evaluate mode: perform validation on PGD attacks as well as clean samples
    if configs.evaluate:
        logger.info(pad_str(' Performing PGD Attacks '))
        for pgd_param in configs.ADV.pgd_attack:
            validate_pgd(val_loader, model, criterion, pgd_param[0], pgd_param[1], configs, logger)
        validate(val_loader, model, criterion, configs, logger)
        return
    

    # calculate the gradients w.r.t conv and bn layers
    if configs.print_gradients == True:
        pattern_bn = re.compile(r'bn')
        pattern_conv = re.compile(r'conv')
        for name, param in model.named_parameters():
            if pattern_bn.search(name) or pattern_conv.search(name):
                name_all.append(name)
                para_all[name] = param
                mean_per_epoch_cle[name] = []
                mean_per_epoch_adv[name] = []
                std_per_epoch_cle[name] = []
                std_per_epoch_adv[name] = []



    
    for epoch in range(configs.TRAIN.start_epoch, configs.TRAIN.epochs):
        adjust_learning_rate(configs.TRAIN.lr, optimizer, epoch, configs.ADV.n_repeats)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, configs, logger)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': configs.TRAIN.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, os.path.join('trained_models', configs.output_name))
        
    # Automatically perform PGD Attacks at the end of training
    logger.info(pad_str(' Performing PGD Attacks '))
    for pgd_param in configs.ADV.pgd_attack:
        validate_pgd(val_loader, model, criterion, pgd_param[0], pgd_param[1], configs, logger)



# Free Adversarial Training Module        
global global_noise_data
global_noise_data = torch.zeros([configs.DATA.batch_size, 3, configs.DATA.crop_size, configs.DATA.crop_size]).cuda()
def train(train_loader, model, criterion, optimizer, epoch):
    global global_noise_data
    global name_all
    global para_all
    global mean_per_epoch_cle
    global std_per_epoch_cle
    global mean_per_epoch_adv
    global std_per_epoch_adv

    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3,configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()


    # calculate the gradients w.r.t conv and bn layers
    if configs.print_gradients == True:
        #initialize count records in each epoch
        count_cle = {}
        count_adv = {}
        for h in name_all:
            count_cle[h] = []
            count_adv[h] = []


    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        onehot_target_withmargin_HE = angular_margin_HE * torch.nn.functional.one_hot(target, num_classes=1000)
         
        data_time.update(time.time() - end)


        for j in range(configs.ADV.n_repeats):
            # Ascend on the global noise
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).cuda()
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            in1.sub_(mean).div_(std)
            output = model(in1)


            # Cosface loss
            if configs.FN == True or configs.WN == True:
                output_withmargin = s_HE * (output - onehot_target_withmargin_HE)
                loss = criterion(output_withmargin, target)
            else:
                loss = criterion(output, target)

            
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            
            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, configs.ADV.fgsm_step)
            global_noise_data[0:input.size(0)] += pert.data
            global_noise_data.clamp_(-configs.ADV.clip_eps, configs.ADV.clip_eps)

            optimizer.step()



            # calculate the gradients w.r.t conv and bn layers
            if configs.print_gradients == True:
                #for adv inputs
                for h in name_all:
                    p = para_all[h].grad
                    count_adv[h].append(torch.norm(p))
                #for clean inputs
                model.zero_grad()
                clean_output = model(input)
                # Cosface loss
                if configs.FN == True or configs.WN == True:
                    clean_output_withmargin = s_HE * (clean_output - onehot_target_withmargin_HE)
                    clean_loss = criterion(clean_output_withmargin, target)
                else:
                    clean_loss = criterion(clean_output, target)
                clean_loss.backward()
                for h in name_all:
                    p = para_all[h].grad
                    count_cle[h].append(torch.norm(p))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % configs.TRAIN.print_freq == 0:
                print('Train Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, top1=top1, top5=top5,cls_loss=losses))
                sys.stdout.flush()


    for h in name_all:
        count_cle_h = torch.tensor(count_cle[h])
        count_adv_h = torch.tensor(count_adv[h])
        mean_per_epoch_cle[h].append(round(torch.mean(count_cle_h).item(),5))
        mean_per_epoch_adv[h].append(round(torch.mean(count_adv_h).item(),5))
        std_per_epoch_cle[h].append(round(torch.std(count_cle_h).item(),5))
        std_per_epoch_adv[h].append(round(torch.std(count_adv_h).item(),5))
    print('mean_cle:', mean_per_epoch_cle)
    print('mean_adv:', mean_per_epoch_adv)
    print('std_cle:', std_per_epoch_cle)
    print('std_adv:', std_per_epoch_adv)

if __name__ == '__main__':
    main()
