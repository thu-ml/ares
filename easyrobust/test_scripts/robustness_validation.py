import os

import argparse
import numpy as np
import time
import json
import yaml
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo

# torchvision func
from torchvision import transforms, datasets

# timm func
from timm.models import create_model
from timm.utils import AverageMeter, reduce_tensor, accuracy

# EasyRobust training func
from utils import NormalizeByChannelMeanStd, robust_model_urls
from metrics import *

# img misc
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image

def get_args_parser():
    parser = argparse.ArgumentParser('Robustness validation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)

    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--num-classes', default=1000, type=int,
                        help='number of classes')

    parser.add_argument('--crop-pct', default=0.875, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--interpolation', default=3, type=int,
                    help='1: lanczos 2: bilinear 3: bicubic')
    parser.add_argument('--mean', type=float, nargs='+', default=(0.485, 0.456, 0.406), metavar='MEAN',
                    help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=(0.229, 0.224, 0.225), metavar='STD',
                        help='Override std deviation of of dataset')

    parser.add_argument('--device', default='cuda',
                        help='cuda | cpu')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')

    # evaluated datasets
    parser.add_argument('--imagenet_val_path', default='', type=str, help='path to imagenet validation dataset')
    parser.add_argument('--imagenet_c_path', default='', type=str, help='path to imagenet curruption dataset')
    parser.add_argument('--imagenet_a_path', default='', type=str, help='path to imagenet natural adversarial dataset')
    parser.add_argument('--imagenet_r_path', default='', type=str, help='path to imagenet rendition dataset')
    parser.add_argument('--imagenet_sketch_path', default='', type=str, help='path to imagenet sketch dataset')
    parser.add_argument('--imagenet_v2_path', default='', type=str, help='path to imagenetv2 dataset')
    parser.add_argument('--stylized_imagenet_path', default='', type=str, help='path to stylized imagenet dataset')
    parser.add_argument('--objectnet_path', default='', type=str, help='path to objectnet dataset')
    parser.add_argument('--whitebox_adv', action='store_true', default=False, help='autoattack, pgd-100')
    
    return parser

def main(args):
    print(args)

    t = []
    if args.input_size > 32:
        size = int(args.input_size/args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=args.interpolation),
        )
        t.append(transforms.CenterCrop(args.input_size))
    else:
        t.append(
            transforms.Resize(args.input_size, interpolation=args.interpolation),
        )
    t.append(transforms.ToTensor())
    test_transform = transforms.Compose(t)

    print('test transforms: ', test_transform)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes
    )

    # auto resume from the robust model urls. If you do not want to resume, comment following lines
    model.load_state_dict(model_zoo.load_url(robust_model_urls[args.model]))
    
    normalize = NormalizeByChannelMeanStd(mean=args.mean, std=args.std)
    model = torch.nn.Sequential(normalize, model)
    model.to(args.device)
    if args.device == 'cuda':
        model = torch.nn.DataParallel(model)
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    print('=' * 30)

    if args.imagenet_val_path:

        # imagenet val (no fix label)
        dataset_val = datasets.ImageFolder(args.imagenet_val_path, transform=test_transform)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=None,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_val, args, log_suffix='ImageNet-Val')
        print(f"ImageNet-Val Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

        if args.whitebox_adv:
            dataset_adv = ImageDataset_Adv(args.imagenet_val_path, transform=test_transform)
            data_loader_adv = torch.utils.data.DataLoader(
            dataset_adv, sampler=None,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
            # val_metrics = adv_validate(model, data_loader_adv, args, log_suffix='PGD100', attack_type='pgd100')
            val_metrics = adv_validate(model, data_loader_adv, args, log_suffix='AutoAttack', attack_type='autoattack')

        # imagenet val (fix label)
        dataset_real = ImageFolderReturnsPath(args.imagenet_val_path, transform=test_transform)
                
        data_loader_real = torch.utils.data.DataLoader(
            dataset_real, sampler=None,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate_for_in_real(model, data_loader_real, args, log_suffix='ImageNet-Real')
        print(f"ImageNet-Real Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

    if args.objectnet_path:
        # objectnet transform
        objectnet_transform = transforms.Compose([transforms.Resize(args.input_size, interpolation=args.interpolation),
                            transforms.CenterCrop(args.input_size),
                            transforms.ToTensor()])

        dataset_objnet = ObjectNetDataset(args.objectnet_path, transform=objectnet_transform)
        data_loader_objnet = torch.utils.data.DataLoader(
            dataset_objnet, sampler=None,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_objnet, args, log_suffix='ObjectNet', mapping=True)
        print(f"ObjectNet Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

    if args.stylized_imagenet_path:
        if args.input_size == 224:
            stylized_in_transform = transforms.Compose([transforms.ToTensor()])
        else:
            stylized_in_transform = transforms.Compose([transforms.Resize(args.input_size, interpolation=args.interpolation),
                        transforms.CenterCrop(args.input_size),
                        transforms.ToTensor()])

        dataset_stylized_in = datasets.ImageFolder(args.stylized_imagenet_path, transform=stylized_in_transform)
        data_loader_stylized_in = torch.utils.data.DataLoader(
            dataset_stylized_in, sampler=None,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_stylized_in, args, log_suffix='Stylized-ImageNet')
        print(f"Stylized-ImageNet Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

    if args.imagenet_v2_path:
        dataset_val_v2 = datasets.ImageFolder(args.imagenet_v2_path, transform=test_transform)
        data_loader_val_v2 = torch.utils.data.DataLoader(
            dataset_val_v2, sampler=None,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_val_v2, args, log_suffix='ImageNet-V2')
        print(f"ImageNet-V2 Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

    if args.imagenet_r_path:
        dataset_inr = datasets.ImageFolder(args.imagenet_r_path, transform=test_transform)
        data_loader_inr = torch.utils.data.DataLoader(
            dataset_inr, sampler=None,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_inr, args, log_suffix='ImageNet-R', mask=imagenet_r_mask)
        print(f"ImageNet-R Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")
    
    if args.imagenet_a_path:
        dataset_ina = datasets.ImageFolder(args.imagenet_a_path, transform=test_transform)
        data_loader_ina = torch.utils.data.DataLoader(
            dataset_ina, sampler=None,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_ina, args, log_suffix='ImageNet-A', mask=imagenet_a_mask)
        print(f"ImageNet-A Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

    if args.imagenet_sketch_path:
        dataset_insk = datasets.ImageFolder(args.imagenet_sketch_path, transform=test_transform)
        data_loader_insk = torch.utils.data.DataLoader(
            dataset_insk, sampler=None,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        val_metrics = validate(model, data_loader_insk, args, log_suffix='ImageNet-Sketch')
        print(f"ImageNet-Sketch Top-1 accuracy of the model is: {val_metrics['top1']:.2f}%")

    if args.imagenet_c_path:
        if not os.path.exists(args.imagenet_c_path):
            print('{} is not exist. skip')
        else:
            result_dict = {}
            ce_alexnet = get_ce_alexnet()

            # transform for imagenet-c
            if args.input_size == 224:
                inc_transform = transforms.Compose([transforms.ToTensor()])
            else:
                inc_transform = transforms.Compose([transforms.Resize(args.input_size, interpolation=args.interpolation),
                            transforms.CenterCrop(args.input_size),
                            transforms.ToTensor()])

            for name, subdir in data_loaders_names.items():
                for severity in range(1, 6):
                    inc_dataset = datasets.ImageFolder(os.path.join(args.imagenet_c_path, subdir, str(severity)), transform=inc_transform)
                    inc_data_loader = torch.utils.data.DataLoader(
                                    inc_dataset, sampler=None,
                                    batch_size=args.batch_size,
                                    num_workers=4,
                                    pin_memory=True,
                                    drop_last=False
                                )
                    test_stats = validate(model, inc_data_loader, args, log_suffix='ImageNet-C')
                    print(f"Accuracy on the {name+'({})'.format(severity)}: {test_stats['top1']:.1f}%")
                    result_dict[name+'({})'.format(severity)] = test_stats['top1']

            mCE = 0
            counter = 0
            overall_acc = 0
            for name, _ in data_loaders_names.items():
                acc_top1 = 0
                for severity in range(1, 6):
                    acc_top1 += result_dict[name+'({})'.format(severity)]
                acc_top1 /= 5
                CE = get_mce_from_accuracy(acc_top1, ce_alexnet[name])
                mCE += CE
                overall_acc += acc_top1
                counter += 1
                print("{0}: Top1 accuracy {1:.2f}, CE: {2:.2f}".format(
                        name, acc_top1, 100. * CE))
            
            overall_acc /= counter
            mCE /= counter
            print("Corruption Top1 accuracy {0:.2f}, mCE: {1:.2f}".format(overall_acc, mCE * 100.))

def validate(model, loader, args, log_suffix='', mask=None, mapping=False):
    batch_time_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, target) in enumerate(loader):
        input = input.to(args.device)
        target = target.to(args.device)
        with torch.no_grad():
            output = model(input)

        if mapping:
            _, prediction_class = output.topk(5, 1, True, True)
            prediction_class = prediction_class.data.cpu().tolist()
            for i in range(output.size(0)):
                imageNetIDToObjectNetID(prediction_class[i])

            prediction_class = torch.tensor(prediction_class).to(args.device)
            prediction_class = prediction_class.t()
            correct = prediction_class.eq(target.reshape(1, -1).expand_as(prediction_class))
            acc1, acc5 = [correct[:k].reshape(-1).float().sum(0) * 100. / output.size(0) for k in (1, 5)]

        else:
            if mask is None:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
            else:
                acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))

        top1_m.update(acc1.item(), output.size(0))
        top5_m.update(acc5.item(), output.size(0))

        batch_time_m.update(time.time() - end)
        end = time.time()

        if batch_idx == last_idx or batch_idx % args.log_interval == 0:
            log_name = 'Test ' + log_suffix
            print(
                '{0}: [{1:>4d}/{2}]  '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '.format(
                    log_name, batch_idx, last_idx, batch_time=batch_time_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics

def adv_validate(model, loader, args, attack_type, log_suffix=''):
    batch_time_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()
    if attack_type == 'autoattack':
        from autoattack import AutoAttack
        adversary = AutoAttack(model, norm='Linf', eps=4/255, version='standard')
    elif attack_type == 'pgd100':
        from utils import pgd_attack

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, target) in enumerate(loader):
        input = input.to(args.device)
        target = target.to(args.device)

        if attack_type == 'autoattack':
            x_adv = adversary.run_standard_evaluation(input, target, bs=target.size(0))
        elif attack_type == 'pgd100':
            x_adv = pgd_attack(input, target, model, 4/255, 100, 1/255, 10, device=args.device)

        with torch.no_grad():
            output = model(x_adv.detach())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1_m.update(acc1.item(), output.size(0))
        top5_m.update(acc5.item(), output.size(0))

        batch_time_m.update(time.time() - end)
        end = time.time()
        if batch_idx == last_idx or batch_idx % args.log_interval == 0:
            log_name = 'Test ' + log_suffix
            print(
                '{0}: [{1:>4d}/{2}]  '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '.format(
                    log_name, batch_idx, last_idx, batch_time=batch_time_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics

def validate_for_in_real(model, loader, args, log_suffix=''):
    batch_time_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, img_paths) in enumerate(loader):

        input = input.to(args.device)
        with torch.no_grad():
            output = model(input)

        is_correct = {k: [] for k in (1, 5)}

        _, pred_batch = output.topk(5, 1, True, True)

        pred_batch = pred_batch.cpu().numpy()
        sample_idx = 0
        for pred in pred_batch:
            filename = os.path.basename(img_paths[sample_idx])
            if real_labels[filename]:
                for k in (1, 5):
                    is_correct[k].append(
                        any([p in real_labels[filename] for p in pred[:k]]))
            sample_idx += 1

        acc1 = torch.tensor(float(np.mean(is_correct[1])) * 100.).to(args.device)
        acc5 = torch.tensor(float(np.mean(is_correct[5])) * 100.).to(args.device)

        top1_m.update(acc1.item(), output.size(0))
        top5_m.update(acc5.item(), output.size(0))

        batch_time_m.update(time.time() - end)
        end = time.time()
        if batch_idx == last_idx or batch_idx % args.log_interval == 0:
            log_name = 'Test ' + log_suffix
            print(
                '{0}: [{1:>4d}/{2}]  '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '.format(
                    log_name, batch_idx, last_idx, batch_time=batch_time_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Robust test script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)