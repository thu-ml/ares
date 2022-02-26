import os
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import yaml
from contextlib import suppress
from collections import OrderedDict

# torchvision func
import torchvision
from torchvision import transforms

# timm func
from timm.data import Mixup, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms_factory import create_transform
from timm.models import create_model, model_parameters, resume_checkpoint, load_checkpoint, convert_splitbn_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, distribute_bn, AverageMeter, reduce_tensor, dispatch_clip_grad, accuracy, get_outdir, CheckpointSaver, update_summary

# EasyRobust training func
from easyrobust.models.layers import convert_switchablebn_model, convert_padain_model, cn_op_2ins_space_chan
from easyrobust.activation import LP_ReLU2
from easyrobust.data import StyleTransfer
from easyrobust.utils import NormalizeByChannelMeanStd, Lighting, adv_generator, adv_generator_random_target
import easyrobust.models

# img misc
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    from timm.utils import ApexScaler
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass



def get_args_parser():
    parser = argparse.ArgumentParser('Robust training script', add_help=False)
    parser.add_argument('--configs', default='', type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=150, type=int)

    # local test parameters
    parser.add_argument('--local_rank', type=int, default=0)

    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true', default=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.9998, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    parser.add_argument('--pretrained', default='',
                    help='Start with pretrained version of specified network (if avail)')

    # Dataset parameters
    parser.add_argument('--data-dir', default='/aisecurity-group-ds/common_data/deepfake', type=str,
                        help='dataset path')
    parser.add_argument('--num-classes', default=1000, type=int,
                        help='number of classes')
    parser.add_argument('--crop-pct', default=0.875, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--mean', type=float, nargs='+', default=(0.485, 0.456, 0.406), metavar='MEAN',
                    help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=(0.229, 0.224, 0.225), metavar='STD',
                        help='Override std deviation of of dataset')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist-bn', type=str, default='reduce',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')

    # amp parameters
    parser.add_argument('--no_amp', action='store_true', help='not using amp')
    parser.add_argument('--use_apex_amp', action='store_true', help='use apex amp')

    # EasyRobust training parameters
    parser.add_argument('--training_mode', default='regular', choices=['regular', 'advtrain', 'advprop', 'padain', 'debiased', 'crossnorm', 'pyramid_advtrain'])
    # default args for some specific training mode
    parser.add_argument('--padain', default=0.01, type=float, help='default args for: Permuted AdaIN: Reducing the Bias Towards Global Statistics in Image Classification')
    parser.add_argument('--crossnorm', default=0.5, type=float, help='default args for: CrossNorm and SelfNorm for Generalization under Distribution Shifts')
    parser.add_argument('--attack_criterion', type=str, default='regular', choices=['regular', 'smooth', 'mixup'], help='default args for: adversarial training')
    parser.add_argument('--standard_dataaug_for_advtrain', action='store_true', help='default args for: adversarial training')

    return parser

def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

def main(args, args_text):
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args.gpu = int(os.environ['LOCAL_RANK'])
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    print(args)

    device = torch.device(args.device)

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if not args.no_amp:  # args.amp: Default  use AMP
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        else:
            args.native_amp = False
        if has_apex:
            args.apex_amp = True
        else:
            args.apex_amp = False

        if args.native_amp == False and args.apex_amp == False:
            raise ValueError("Warning: Neither APEX or native Torch AMP is available, using float32."
                             "Install NVIDA apex or upgrade to PyTorch 1.6")
    else:
        args.apex_amp = False
        args.native_amp = False
        
    if args.apex_amp and args.use_apex_amp:
        use_amp = 'apex'
    elif args.native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        print ("Warning: Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    # fix the seed for reproducibility
    seed = args.seed + args.rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    # build data transform
    train_transform = create_transform(
        input_size=args.input_size,
        is_training=True,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=args.train_interpolation,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        mean=args.mean,
        std=args.std,
        crop_pct=args.crop_pct
    )
    if not args.input_size > 32:
        # replace RandomResizedCropAndInterpolation with
        # RandomCrop
        train_transform.transforms[0] = transforms.RandomCrop(
            args.input_size, padding=4)

    if args.training_mode in ['advtrain', 'pyramid_advtrain', 'advprop']:
        # denormalize the input
        train_transform.transforms.append(transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            ))
        if args.standard_dataaug_for_advtrain:
                train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1
                ),
                transforms.ToTensor(),
                Lighting(0.05, torch.Tensor([0.2175, 0.0188, 0.0045]), 
                            torch.Tensor([
                [-0.5675,  0.7192,  0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948,  0.4203],
            ])),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    t = []
    if args.input_size > 32:
        size = int(args.input_size/args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.input_size))
    else:
        t.append(
            transforms.Resize(args.input_size, interpolation=3),
        )
    t.append(transforms.ToTensor())
    if args.training_mode not in ['advtrain', 'advprop', 'pyramid_advtrain']:
        t.append(transforms.Normalize(args.mean, args.std))
    test_transform = transforms.Compose(t)

    print('train transforms: ', train_transform)
    print('test transforms: ', test_transform)

    dataset_train = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transform)
    dataset_val = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform=test_transform)

    sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True
            )
    sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=args.world_size, rank=args.rank, shuffle=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None
    )
    # style transfer augment
    style_transfer = None
    if args.training_mode == 'advprop':
        model = convert_switchablebn_model(model)
    elif args.training_mode == 'debiased':
        model = convert_splitbn_model(model)
        style_transfer = StyleTransfer(mean=torch.tensor([0.485, 0.456, 0.406]).cuda(), std=torch.tensor([0.229, 0.224, 0.225]).cuda())
    elif args.training_mode == 'padain':
        model = convert_padain_model(model, padain=args.padain)

    if args.pretrained:
        if args.pretrained.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrained, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.pretrained, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        model.load_state_dict(checkpoint_model, strict=False)

    if args.training_mode in ['advtrain', 'advprop', 'pyramid_advtrain']:
        normalize = NormalizeByChannelMeanStd(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        model = torch.nn.Sequential(normalize, model)
    model.to(device)

    # linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    # args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        lr_scheduler, _ = create_scheduler(args, optimizer)

        loss_scaler = ApexScaler()
        print('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        print('Using native Torch AMP. Training in mixed precision.')
    else:
        print('AMP not enabled. Training in float32.')

    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            log_info=args.rank == 0)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)

    model_without_ddp = model
    if args.distributed:
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            model = ApexDDP(model, delay_allreduce=True)
            print("Using NVIDIA APEX DistributedDataParallel.")
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            print("Using native Torch DistributedDataParallel.")
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    print('=' * 30)

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)
    print('Scheduled epochs: {}'.format(num_epochs))

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0. or args.training_mode == 'debiased':
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.eval:
        load_checkpoint(model, args.eval_checkpoint, args.model_ema)
        val_metrics = validate(model, data_loader_val, torch.nn.CrossEntropyLoss().cuda(), args)
        print(f"Top-1 accuracy of the model is: {val_metrics['top1']:.1f}%")
        return

    saver = None
    best_metric = None
    best_epoch = None
    output_dir = None
    if args.rank == 0:
        output_dir = get_outdir(args.output_dir)
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=False)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
                epoch, model, data_loader_train, optimizer, criterion, args,
                lr_scheduler=lr_scheduler, saver=saver, amp_autocast=amp_autocast, 
                loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn, style_transfer=style_transfer)

        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
            if args.rank == 0:
                print("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

        eval_metrics = validate(model, data_loader_val, torch.nn.CrossEntropyLoss().cuda(), args)

        if model_ema is not None and not args.model_ema_force_cpu:
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
            ema_eval_metrics = validate(
                model_ema.module, data_loader_val, torch.nn.CrossEntropyLoss().cuda(), args, log_suffix=' (EMA)')
            eval_metrics = ema_eval_metrics

        if lr_scheduler is not None:
            # step LR for next epoch
            lr_scheduler.step(epoch + 1, eval_metrics['top1'])

        if output_dir is not None:
            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

        if saver is not None:
            best_metric, best_epoch = saver.save_checkpoint(epoch, eval_metrics['top1'])
        
        if best_metric is not None:
            print('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None, style_transfer=None):

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        input, target = input.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)
        if style_transfer is not None:
            input, target = style_transfer(input, target, replace=False, alpha=0.5, label_mix_alpha=0.2)

        if args.training_mode == 'crossnorm':
            if np.random.rand(1) < args.crossnorm:
                input = cn_op_2ins_space_chan(input)

        if mixup_fn is not None:
            input, target = mixup_fn(input, target)

        if isinstance(target, tuple):
            target = one_hot(target[0], args.num_classes) * (1-target[2][:,None]) + one_hot(target[1], args.num_classes) * target[2][:,None]

        
        if args.training_mode == 'advtrain':
            input = adv_generator(input, target, model, 4/255, 3, 8/255/3, random_start=False, gpu=args.gpu, attack_criterion=args.attack_criterion)
        elif args.training_mode == 'pyramid_advtrain':
            adv_input = adv_generator_random_target(input, target, model, 4/255, 3, 8/255/3, random_start=True, gpu=args.gpu)
        elif args.training_mode == 'advprop':
            model.apply(lambda m: setattr(m, 'bn_mode', 'adv'))
            adv_input = adv_generator(input, target, model, 1/255, 1, 1/255, random_start=True, gpu=args.gpu, attack_criterion=args.attack_criterion)
        
        with amp_autocast():
            if args.training_mode == 'pyramid_advtrain' or args.training_mode == 'discrete_advtrain':
                loss = loss_fn(model(input), target) + loss_fn(model(adv_input), target)
            elif args.training_mode == 'advprop':
                outputs = model(adv_input)
                adv_loss = loss_fn(outputs, target)
                model.apply(lambda m: setattr(m, 'bn_mode', 'clean'))
                outputs = model(input)
                loss = loss_fn(outputs, target) + adv_loss
            else:
                output = model(input)
                loss = loss_fn(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            print(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'LR: {lr:.3e}  '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) * args.world_size / batch_time_m.val,
                    rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m))

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    adv_top1_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)

        if args.training_mode in ['advtrain', 'advprop', 'pyramid_advtrain']:
            advinput = adv_generator(input, target, model, 4/255, 3, 8/255/3, random_start=True, gpu=args.gpu, use_best=False, attack_criterion='regular')
            with torch.no_grad():
                advoutput = model(advinput)
            acc1, acc5 = accuracy(advoutput, target, topk=(1, 5))
            if args.distributed:
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            torch.cuda.synchronize()
            adv_top1_m.update(acc1.item(), advoutput.size(0))

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            acc1 = reduce_tensor(acc1, args.world_size)
            acc5 = reduce_tensor(acc5, args.world_size)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        losses_m.update(reduced_loss.item(), input.size(0))
        top1_m.update(acc1.item(), output.size(0))
        top5_m.update(acc5.item(), output.size(0))

        batch_time_m.update(time.time() - end)
        end = time.time()

        log_name = 'Test' + log_suffix
        print(
            '{0}: [{1:>4d}/{2}]  '
            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
            'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
            'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
            'Adv: {advtop1.val:>7.4f} ({advtop1.avg:>7.4f})'.format(
                log_name, batch_idx, last_idx, batch_time=batch_time_m,
                loss=losses_m, top1=top1_m, top5=top5_m, advtop1=adv_top1_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Robust training script', parents=[get_args_parser()])
    args = parser.parse_args()
    opt = vars(args)
    if args.configs:
        opt.update(yaml.load(open(args.configs), Loader=yaml.FullLoader))
    
    args = argparse.Namespace(**opt)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    main(args, args_text)