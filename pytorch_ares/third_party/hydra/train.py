# Some part borrowed from official tutorial https://github.com/pytorch/examples/blob/master/imagenet/main.py
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import numpy as np
import argparse
import importlib
import time
import logging
from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import models
import data
from args import parse_args
from utils.schedules import get_lr_policy, get_optimizer
from utils.logging import (
    save_checkpoint,
    create_subdirs,
    parse_configs_file,
    clone_results_to_latest_subdir,
)
from utils.semisup import get_semisup_dataloader
from utils.model import (
    get_layers,
    prepare_model,
    initialize_scaled_score,
    scale_rand_init,
    show_gradients,
    current_model_pruned_fraction,
    sanity_check_paramter_updates,
    snip_init,
)


# TODO: update wrn, resnet models. Save both subnet and dense version.
# TODO: take care of BN, bias in pruning, support structured pruning


def main():
    args = parse_args()
    parse_configs_file(args)

    # sanity checks
    if args.exp_mode in ["prune", "finetune"] and not args.resume:
        assert args.source_net, "Provide checkpoint to prune/finetune"

    # create resutls dir (for logs, checkpoints, etc.)
    result_main_dir = os.path.join(Path(args.result_dir), args.exp_name, args.exp_mode)

    if os.path.exists(result_main_dir):
        n = len(next(os.walk(result_main_dir))[-2])  # prev experiments with same name
        result_sub_dir = os.path.join(
            result_main_dir,
            "{}--k-{:.2f}_trainer-{}_lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                n + 1,
                args.k,
                args.trainer,
                args.lr,
                args.epochs,
                args.warmup_lr,
                args.warmup_epochs,
            ),
        )
    else:
        os.makedirs(result_main_dir, exist_ok=True)
        result_sub_dir = os.path.join(
            result_main_dir,
            "1--k-{:.2f}_trainer-{}_lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                args.k,
                args.trainer,
                args.lr,
                args.epochs,
                args.warmup_lr,
                args.warmup_epochs,
            ),
        )
    create_subdirs(result_sub_dir)

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    )
    logger.info(args)

    # seed cuda
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Select GPUs
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if use_cuda else "cpu")

    # Create model
    cl, ll = get_layers(args.layer_type)
    if len(gpu_list) > 1:
        print("Using multiple GPUs")
        model = nn.DataParallel(
            models.__dict__[args.arch](
                cl, ll, args.init_type, num_classes=args.num_classes
            ),
            gpu_list,
        ).to(device)
    else:
        model = models.__dict__[args.arch](
            cl, ll, args.init_type, num_classes=args.num_classes
        ).to(device)
    logger.info(model)

    # Customize models for training/pruning/fine-tuning
    prepare_model(model, args)

    # Setup tensorboard writer
    writer = SummaryWriter(os.path.join(result_sub_dir, "tensorboard"))

    # Dataloader
    D = data.__dict__[args.dataset](args, normalize=args.normalize)
    train_loader, test_loader = D.data_loaders()

    logger.info(args.dataset, D, len(train_loader.dataset), len(test_loader.dataset))

    # Semi-sup dataloader
    if args.is_semisup:
        logger.info("Using semi-supervised training")
        sm_loader = get_semisup_dataloader(args, D.tr_train)
    else:
        sm_loader = None

    # autograd
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)
    logger.info([criterion, optimizer, lr_policy])

    # train & val method
    trainer = importlib.import_module(f"trainer.{args.trainer}").train
    val = getattr(importlib.import_module("utils.eval"), args.val_method)

    # Load source_net (if checkpoint provided). Only load the state_dict (required for pruning and fine-tuning)
    if args.source_net:
        if os.path.isfile(args.source_net):
            logger.info("=> loading source model from '{}'".format(args.source_net))
            checkpoint = torch.load(args.source_net, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            logger.info("=> loaded checkpoint '{}'".format(args.source_net))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Init scores once source net is loaded.
    # NOTE: scaled_init_scores will overwrite the scores in the pre-trained net.
    if args.scaled_score_init:
        initialize_scaled_score(model)

    # Scaled random initialization. Useful when training a high sparse net from scratch.
    # If not used, a sparse net (without batch-norm) from scratch will not coverge.
    # With batch-norm its not really necessary.
    if args.scale_rand_init:
        scale_rand_init(model, args.k)

    # Scaled random initialization. Useful when training a high sparse net from scratch.
    # If not used, a sparse net (without batch-norm) from scratch will not coverge.
    # With batch-norm its not really necessary.
    if args.scale_rand_init:
        scale_rand_init(model, args.k)

    if args.snip_init:
        snip_init(model, criterion, optimizer, train_loader, device, args)

    assert not (args.source_net and args.resume), (
        "Incorrect setup: "
        "resume => required to resume a previous experiment (loads all parameters)|| "
        "source_net => required to start pruning/fine-tuning from a source model (only load state_dict)"
    )
    # resume (if checkpoint provided). Continue training with preiovus settings.
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Evaluate
    if args.evaluate or args.exp_mode in ["prune", "finetune"]:
        p1, _ = val(model, device, test_loader, criterion, args, writer)
        logger.info(f"Validation accuracy {args.val_method} for source-net: {p1}")
        if args.evaluate:
            return

    best_prec1 = 0

    show_gradients(model)

    if args.source_net:
        last_ckpt = checkpoint["state_dict"]
    else:
        last_ckpt = copy.deepcopy(model.state_dict())

    # Start training
    for epoch in range(args.start_epoch, args.epochs + args.warmup_epochs):
        lr_policy(epoch)  # adjust learning rate

        # train
        trainer(
            model,
            device,
            train_loader,
            sm_loader,
            criterion,
            optimizer,
            epoch,
            args,
            writer,
        )

        # evaluate on test set
        if args.val_method == "smooth":
            prec1, radii = val(
                model, device, test_loader, criterion, args, writer, epoch
            )
            logger.info(f"Epoch {epoch}, mean provable Radii  {radii}")
        if args.val_method == "mixtrain" and epoch <= args.schedule_length:
            prec1 = 0.0
        else:
            prec1, _ = val(model, device, test_loader, criterion, args, writer, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args,
            result_dir=os.path.join(result_sub_dir, "checkpoint"),
            save_dense=args.save_dense,
        )

        logger.info(
            f"Epoch {epoch}, val-method {args.val_method}, validation accuracy {prec1}, best_prec {best_prec1}"
        )
        if args.exp_mode in ["prune", "finetune"]:
            logger.info(
                "Pruned model: {:.2f}%".format(
                    current_model_pruned_fraction(
                        model, os.path.join(result_sub_dir, "checkpoint"), verbose=False
                    )
                )
            )
        # clone results to latest subdir (sync after every epoch)
        # Latest_subdir: stores results from latest run of an experiment.
        clone_results_to_latest_subdir(
            result_sub_dir, os.path.join(result_main_dir, "latest_exp")
        )

        # Check what parameters got updated in the current epoch.
        sw, ss = sanity_check_paramter_updates(model, last_ckpt)
        logger.info(
            f"Sanity check (exp-mode: {args.exp_mode}): Weight update - {sw}, Scores update - {ss}"
        )

    current_model_pruned_fraction(
        model, os.path.join(result_sub_dir, "checkpoint"), verbose=True
    )


if __name__ == "__main__":
    main()
