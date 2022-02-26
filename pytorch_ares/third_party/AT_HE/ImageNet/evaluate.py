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
from validation import validate, validate_pgd, validate_ImagetNet_C
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
    parser.add_argument('--eva_on_imagenet_c', dest='eva_on_imagenet_c', action='store_true',
                    help='evaluate model on imagenet-c')
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
    normalize = transforms.Normalize(mean=configs.TRAIN.mean, std=configs.TRAIN.std)

    # If in evaluate mode: perform validation on PGD attacks as well as clean samples
    if configs.evaluate:
        valdir = os.path.join(configs.data, 'val')
        #fix random seed for save images and perturbation
        #torch.manual_seed(1234)
        #np.random.seed(1234)
        val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(configs.DATA.img_size),
            transforms.CenterCrop(configs.DATA.crop_size),
            transforms.ToTensor(),
        ])),
        batch_size=configs.DATA.batch_size, shuffle=False,
        num_workers=configs.DATA.workers, pin_memory=True)
        if configs.FN or configs.WN:
            print('use HE for evaluate and save images!!!!!!!!!!')
            HEs = True
        else:
            print('DO NOT use HE for evaluate and save images!!!!!!!!!!')
            HEs = False
        logger.info(pad_str(' Performing PGD Attacks '))
        for pgd_param in configs.ADV.pgd_attack:
            validate_pgd(val_loader, model, criterion, pgd_param[0], pgd_param[1], configs, logger, save_image=True, HE=HEs)
        validate(val_loader, model, criterion, configs, logger)
        return


    
    # If evaluate on ImageNet-C
    if configs.eva_on_imagenet_c:
        if configs.FN or configs.WN:
            print('use HE for evaluate and save images!!!!!!!!!!')
        else:
            print('DO NOT use HE for evaluate and save images!!!!!!!!!!')
        logger.info(pad_str(' Performing evaluation on ImageNet-C'))

        files_names = ['gaussian_noise', 'shot_noise', 'impulse_noise',#noise
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', #blur
        'frost', 'snow', 'fog', 'brightness', #whether
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']#digital

        for file_n in files_names:
            print('Processing: ', file_n)
            validate_ImagetNet_C(file_n, model, criterion, configs, logger)
        return


if __name__ == '__main__':
    main()
