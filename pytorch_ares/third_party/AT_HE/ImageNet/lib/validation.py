from utils import *
import torch
import sys
import numpy as np
import time
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def validate_pgd(val_loader, model, criterion, K, step, configs, logger, save_image=False, HE=False):
    # Mean/Std for normalization   
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3,configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    # Initiate the meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    eps = configs.ADV.clip_eps
    model.eval()
    end = time.time()
    logger.info(pad_str(' PGD eps: {}, K: {}, step: {} '.format(eps, K, step)))

    if HE == True:
        is_HE = '_HE'
    else:
        is_HE = ''
    if configs.pretrained:
        is_HE = '_pretrained'

    for i, (input, target) in enumerate(val_loader):
        
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        #save original images
        if save_image == True and i < 2:
            original_images_save = input.clone()
            for o in range(input.size(0)):
                torchvision.utils.save_image(original_images_save[o, :, :, :], 'saved_images/original_images'+is_HE+'/{}.png'.format(o + configs.DATA.batch_size*i))

        

        randn = torch.FloatTensor(input.size()).uniform_(-eps, eps).cuda()
        input += randn
        input.clamp_(0, 1.0)

        orig_input = input.clone()
        for _ in range(K):
            invar = Variable(input, requires_grad=True)
            in1 = invar - mean
            in1.div_(std)
            output = model(in1)
            ascend_loss = criterion(output, target)
            ascend_grad = torch.autograd.grad(ascend_loss, invar)[0]
            pert = fgsm(ascend_grad, step)
            # Apply purturbation
            input += pert.data
            input = torch.max(orig_input-eps, input)
            input = torch.min(orig_input+eps, input)
            input.clamp_(0, 1.0)
        
        #save adv images
        if save_image == True and i < 2:
            adv_images_save = input.clone()
            for o in range(input.size(0)):
                torchvision.utils.save_image(adv_images_save[o, :, :, :], 'saved_images/adv_images'+is_HE+'/{}.png'.format(o + configs.DATA.batch_size*i))
        



        #save scaled perturbation
        perturbation = input - orig_input
        perturbation.clamp_(-eps,eps)
        scaled_perturbation = (perturbation.clone() + eps) / (2 * eps)
        scaled_perturbation.clamp_(0, 1.0)
        if save_image == True and i < 2:
            for o in range(input.size(0)):
                torchvision.utils.save_image(scaled_perturbation[o, :, :, :], 'saved_images/scaled_perturbation'+is_HE+'/{}.png'.format(o + configs.DATA.batch_size*i))


        input.sub_(mean).div_(std)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % configs.TRAIN.print_freq == 0:
                print('PGD Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
                sys.stdout.flush()

    print(' PGD Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return top1.avg

def validate(val_loader, model, criterion, configs, logger):
    # Mean/Std for normalization   
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3,configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    
    # Initiate the meters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            input = input - mean
            input.div_(std)
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % configs.TRAIN.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
                sys.stdout.flush()

    print(' Final Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))
    return top1.avg


def validate_ImagetNet_C(val_loader_name, model, criterion, configs, logger):
    # Mean/Std for normalization   
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3,configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    
    
    # switch to evaluate mode
    model.eval()

    fil_index = ['/1','/2','/3','/4','/5']

    avg_return = 0

    for f in fil_index:
        valdir = os.path.join(configs.data, val_loader_name+f)
        print(' File: ', valdir)
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(configs.DATA.img_size),
                transforms.CenterCrop(configs.DATA.crop_size),
                transforms.ToTensor(),
            ])),
            batch_size=configs.DATA.batch_size, shuffle=False,
            num_workers=configs.DATA.workers, pin_memory=True)


        # Initiate the meters
        top1 = AverageMeter()
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            with torch.no_grad():
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # compute output
                input = input - mean
                input.div_(std)
                output = model(input)

                # measure accuracy and record loss
                prec1,_ = accuracy(output, target, topk=(1,2))
                top1.update(prec1[0], input.size(0))

                # if i % configs.TRAIN.print_freq == 0:
                #     print('PGD Test: [{0}/{1}]\t'
                #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                #            i, len(val_loader),top1=top1))
                #     print('Time: ', time.time() - end)
                #     sys.stdout.flush()

        print('Prec: ',top1.avg.cpu().item())
        avg_return += top1.avg.cpu().item()
    print('Avergae Classification Accuracy is: ', avg_return / 5.)
    return




