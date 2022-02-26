# evaluate a smoothed classifier on a dataset
import argparse
import os
import sys
import numpy as np
from time import time
import datetime
import importlib
import logging

import torch
import torch.nn as nn

import models
import data

from utils.smoothing import Smooth, quick_smoothing, eval_quick_smoothing
from utils.model import get_layers

parser = argparse.ArgumentParser(description="Certify many examples")
parser.add_argument(
    "--dataset",
    type=str,
    choices=("CIFAR10", "CIFAR100", "SVHN", "MNIST", "imagenet"),
    help="Dataset for training and eval",
)
parser.add_argument(
    "--normalize",
    action="store_true",
    default=False,
    help="whether to normalize the data",
)
parser.add_argument(
    "--data-dir", type=str, default="./datasets", help="path to datasets"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for training (default: 128)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size for testing (default: 128)",
)
parser.add_argument(
    "--data-fraction",
    type=float,
    default=1.0,
    help="Fraction of images used from training set",
)

parser.add_argument(
    "--base_classifier", type=str, help="path to saved pytorch model of base classifier"
)
parser.add_argument("--arch", type=str, default="vgg16_bn", help="Model achitecture")
parser.add_argument(
    "--num-classes", type=int, default=10, help="Number of output classes in the model",
)
parser.add_argument(
    "--layer-type", type=str, choices=("dense", "subnet"), help="dense | subnet layers"
)
parser.add_argument(
    "--noise_std", type=float, default=0.25, help="noise hyperparameter"
)
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")

# parser.add_argument(
#    "--split", choices=["train", "test"], default="test", help="train or test set"
# )
parser.add_argument(
    "--gpu", type=str, default="0", help="Comma separated list of GPU ids"
)
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument(
    "--print-freq",
    type=int,
    default=100,
    help="Number of batches to wait before printing training logs",
)
args = parser.parse_args()

if __name__ == "__main__":

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(args.outfile + ".log", "a"))
    logger.info(args)

    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}")

    # Create model
    cl, ll = get_layers(args.layer_type)
    if len(gpu_list) > 1:
        print("Using multiple GPUs")
        base_classifier = nn.DataParallel(
            models.__dict__[args.arch](
                cl, ll, "kaiming_normal", num_classes=args.num_classes
            ),
            gpu_list,
        ).to(device)
    else:
        base_classifier = models.__dict__[args.arch](
            cl, ll, "kaiming_normal", num_classes=args.num_classes
        ).to(device)

    checkpoint = torch.load(args.base_classifier, map_location=device)
    base_classifier.load_state_dict(checkpoint["state_dict"])

    smoothed_classifier = Smooth(base_classifier, args.num_classes, args.noise_std)

    # prepare output file
    f = open(args.outfile, "w")
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # Dataset
    D = data.__dict__[args.dataset](args, normalize=args.normalize)
    train_loader, test_loader = D.data_loaders()
    dataset = test_loader.dataset  # Certify test inputs only (default)

    val = getattr(importlib.import_module("utils.eval"), "smooth")
    p, r = val(base_classifier, device, test_loader, nn.CrossEntropyLoss(), args, None)
    logger.info(f"Validation natural accuracy for source-net: {p}, radisu: {r}")

    # for i, v in base_classifier.named_modules():
    #     if isinstance(v, (nn.BatchNorm2d, nn.BatchNorm1d)):
    #         v.track_running_stats = False

    # eval_quick_smoothing(base_classifier, train_loader, device, sigma=0.25, nbatch=10)

    base_classifier.eval()
    # sys.exit()

    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.to(device)
        prediction, radius = smoothed_classifier.certify(
            x, args.N0, args.N, args.alpha, args.batch_size, device
        )
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print(
            "{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed
            ),
            file=f,
            flush=True,
        )

    f.close()
