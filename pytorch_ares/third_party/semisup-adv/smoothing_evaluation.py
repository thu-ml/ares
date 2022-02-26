"""
Run randomized certification on the test set. Loosely based on code from
https://github.com/locuslab/smoothing
"""

import argparse
import os
from time import time

import datetime
from utils import get_model

import logging
import pandas as pd

import torch
import torch.nn
import torch.nn.functional as F
from datasets import SemiSupervisedDataset, DATASETS
from torchvision import datasets, transforms

from smoothing import Smooth

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Randomized smoothing certification')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=DATASETS,
                        help='The dataset')
    parser.add_argument('--data_dir', default='data', type=str,
                        help='Directory where datasets are located')
    parser.add_argument('--model_path',
                        help='Model for certification')
    parser.add_argument('--model', '-m', default='wrn-28-10', type=str,
                        help='Name of the model')
    parser.add_argument('--output_name', default='smoothing', type=str,
                        help='Name of output files')
    parser.add_argument("--sigma", default=0.25, type=float,
                        help="Noise hyperparameter")
    parser.add_argument("--batch", type=int, default=1000, help="Batch size")
    parser.add_argument("--skip", type=int, default=1,
                        help="Skip this many examples between each examples "
                             "we certify")
    parser.add_argument("--max", type=int, default=-1,
                        help="Stop when example index == max")
    # parser.add_argument("--split", choices=["train", "test"], default="test",
    #                     help="train or test set")
    parser.add_argument("--N0", type=int, default=100,
                        help="Number of noise sample for classification "
                             "decision")
    parser.add_argument("--N", type=int, default=10000,
                        help="Number of noise sample for radius evaluation")
    parser.add_argument("--alpha", type=float, default=0.001,
                        help="Failure probability")
    parser.add_argument('--random_seed', type=int, default=None)

    args = parser.parse_args()

    # create log
    output_dir, _ = os.path.split(args.model_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir,
                                             '%s.log' % args.output_name)),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()

    logging.info('Randomized smoothing certification')
    logging.info('Args: %s', args)

    # also store stuff in data frame
    df = pd.DataFrame()

    # load the base classifier
    checkpoint = torch.load(args.model_path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    if not all(k.startswith('module') for k in state_dict):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    num_classes = checkpoint.get('num_classes', 10)
    normalize_input = checkpoint.get('normalize_input', False)

    base_classifier = get_model(args.model, num_classes=num_classes,
                                normalize_input=normalize_input)
    base_classifier = torch.nn.DataParallel(base_classifier).cuda()
    # setting loader to be non-strict so we can load Cohen et al.'s model
    base_classifier.load_state_dict(state_dict,
                                    strict=(args.model != 'resnet-110'))

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, num_classes, args.sigma)

    # iterate through the dataset
    transform_test = transforms.ToTensor()
    # dataset = datasets.CIFAR10(root='data', train=False,
    #                            download=True,
    #                            transform=transform_test)
    dataset = SemiSupervisedDataset(base_dataset=args.dataset,
                                    train=False, root=args.data_dir,
                                    download=True,
                                    transform=transform_test)

    # Shuffling the dataset if random seed is not None
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        np.random.shuffle(dataset.targets)
        np.random.seed(args.random_seed)
        np.random.shuffle(dataset.data)
        filename = args.output_name + '_seed_' + str(args.random_seed) + '.csv'
    else:
        filename = args.output_name + '.csv'

    if os.path.exists(os.path.join(output_dir, filename)):
        logging.info('Output file exists, resuming...')
        df = pd.read_csv(os.path.join(output_dir, filename), index_col=0)
        i_start = int(df.i.values[-1]) + 1
        is_correct = list(df.correct.values)
        is_rob_correct = list(df.correct.values *
                              (df.radius.values >= args.sigma))
    else:
        i_start = 0
        is_correct = []
        is_rob_correct = []

    for i in range(i_start, len(dataset)):
        # only certify every args.skip examples, and stop
        # after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, pAbar, radius, counts = smoothed_classifier.certify(
            x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)
        is_correct.append(correct)
        is_rob_correct.append(correct * (radius >= args.sigma))

        time_elapsed = str(
            datetime.timedelta(seconds=(after_time - before_time)))
        logging.info("{}/{} | correct={}, pAbar={:.5f}, radius={:.3f}, "
                     "clean accuracy={:.1f}%, robust accuracy={:.1f}%".format(
            i + 1, len(dataset), correct, pAbar, radius,
            100 * np.mean(is_correct), 100 * np.mean(is_rob_correct)))

        df = df.append(pd.Series(dict(i=i, label=label, prediction=prediction,
                                      pAbar=pAbar, radius=radius,
                                      correct=correct, counts=counts,
                                      time_elapsed=time_elapsed)),
                       ignore_index=True)
        df.to_csv(os.path.join(output_dir, filename))


