import torch
from torchvision import transforms
import math
import argparse
from ares.utils.registry import registry
from ares.utils.metrics import AverageMeter, accuracy
from ares.utils.logger import setup_logger
from ares.dataset.imagenet_dataset import ImageNetDataset
from ares.model import cifar_model_zoo, imagenet_model_zoo
from torchvision.datasets import CIFAR10
from classification.attack_configs import attack_configs

def get_args_parser():
    parser = argparse.ArgumentParser()
    # device
    parser.add_argument("--gpu", type=str, default="0", help="Comma separated list of GPU ids")
    
    # data settings
    parser.add_argument('--crop_pct', type=float, default=0.875, help='Input image center crop percent') 
    parser.add_argument('--input_size', type=int, default=224, help='Input image size') 
    parser.add_argument('--interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic'], help='') 
    parser.add_argument('--data_dir', type=str, default='', help= 'Dataset directory for picture')
    parser.add_argument('--label_file',type=str, default='', help= 'Dataset directory')
    parser.add_argument('--batchsize', type=int, default=10, help= 'batchsize for this model')
    parser.add_argument('--num_workers', type=int, default=8, help= 'number of workers')
    
    # attack and model
    parser.add_argument('--attack_name', type=str, default='tim', help= 'Dataset for this model')
    parser.add_argument('--model_name', type=str, default='resnet50_at', help= 'Model name')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'cifar10'], help= 'ImageNet and cifar10 supported, you need to define custom dataset class if other dataset being used.')
    
    args = parser.parse_args()
    
    return args


def main(args):
    # set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # set logger
    logger = setup_logger()
    
    # create dataloader
    if args.dataset == 'cifar10':
        val_transforms = transforms.Compose([transforms.ToTensor()])
        val_dataset = CIFAR10(args.data_dir, train=False, download=True, transform=val_transforms)
    elif args.dataset == 'imagenet':
        input_resize = int(math.floor(args.input_size / args.crop_pct))
        interpolation_mode={'nearest':0, 'bilinear':2, 'bicubic':3, 'box':4, 'hamming':5, 'lanczos':1}
        interpolation=interpolation_mode[args.interpolation]
        val_transforms = transforms.Compose([transforms.Resize(size=input_resize, interpolation=interpolation),
                                             transforms.CenterCrop(args.input_size),
                                             transforms.ToTensor()])
        val_dataset = ImageNetDataset(args.data_dir, args.label_file, transform=val_transforms)
        
    test_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchsize, num_workers=args.num_workers, 
        shuffle=False, pin_memory=True, drop_last=False)

    # create model
    logger.info('Loading {}...'.format(args.model_name))
    if args.dataset == 'imagenet':
        assert args.model_name in imagenet_model_zoo.keys(), "Model not supported."
        model_cls = registry.get_model('ImageNetCLS')
    else:
        assert args.model_name in cifar_model_zoo.keys(), "Model not supported."
        model_cls = registry.get_model('CifarCLS')
    model = model_cls(args.model_name)
    model = model.to(device)

    # initialize attacker
    attacker_cls = registry.get_attack(args.attack_name)
    attack_config = attack_configs[args.attack_name]
    attacker = attacker_cls(model=model, device=device, **attack_config)

    # attack process
    top1_m = AverageMeter()
    adv_top1_m = AverageMeter()
    for i, (images, labels) in enumerate(test_loader):
        # load data
        batchsize = images.shape[0]
        images, labels = images.to(device), labels.to(device)
        
        # clean acc
        with torch.no_grad():
            logits = model(images)
        clean_acc = accuracy(logits, labels)[0]
        top1_m.update(clean_acc.item(), batchsize)
        
        # robust acc
        adv_images = attacker(images = images, labels = labels, target_labels = None)
        if args.attack_name == 'autoattack':
            if adv_images is None:
                adv_acc = 0.0
            else:
                adv_acc = adv_images.size(0) / batchsize * 100
        else:
            with torch.no_grad():
                adv_logits = model(adv_images)
            adv_acc = accuracy(adv_logits, labels)[0]
            adv_acc = adv_acc.item()
        adv_top1_m.update(adv_acc, batchsize)

    logger.info("Clean accuracy of {0} is {1}%".format(args.model_name, round(top1_m.avg, 2)))
    logger.info("Robust accuracy of {0} is {1}%".format(args.model_name, round(adv_top1_m.avg, 2)))


if __name__ == "__main__":
    args = get_args_parser()
    main(args)

