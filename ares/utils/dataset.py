import torch
from timm.data import Mixup, AugMixDataset, create_transform
from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler
from torchvision import datasets

def build_dataset(args, num_aug_splits=0):
    '''The function to build dataset for robust training.'''
    # build dataset
    dataset_train = datasets.ImageFolder(root=args.train_dir, transform=None)
    dataset_eval = datasets.ImageFolder(root=args.eval_dir, transform=None)
    # dataset_eval=ImageNet(root=args.eval_dir)

    # wrap dataset_train in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # build transform
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = args.interpolation
    re_num_splits = 0
    if args.resplit:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    dataset_train.transform = create_transform(
        args.input_size,
        is_training=True,
        use_prefetcher=False,
        no_aug=args.no_aug,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=train_interpolation,
        mean=args.mean,
        std=args.std,
        crop_pct=args.crop_pct,
        tf_preprocessing=False,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0
    )

    dataset_eval.transform = create_transform(
        args.input_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=args.interpolation,
        mean=args.mean,
        std=args.std,
        crop_pct=args.crop_pct
    )

    # create sampler
    sampler_train = None
    sampler_eval = None
    if args.distributed and not isinstance(dataset_train, torch.utils.data.IterableDataset):
        if args.aug_repeats:
            sampler_train = RepeatAugSampler(dataset_train, num_repeats=args.aug_repeats)
        else:
            sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        assert args.aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"
    sampler_eval = OrderedDistributedSampler(dataset_eval)

    # create dataloader
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=sampler_train,
        collate_fn=None,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    dataloader_eval = torch.utils.data.DataLoader(
        dataset=dataset_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=sampler_eval,
        collate_fn=None,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        mixup_fn = Mixup(**mixup_args)

    return dataloader_train, dataloader_eval, mixup_fn