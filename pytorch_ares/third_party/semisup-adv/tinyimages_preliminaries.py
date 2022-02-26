"""
Preliminary computations for sourcing unlabeled data:
1) Compute the l2 distance between every image in TinyImages and its nearest
neighbor in the CIFAR-10 test set. We will use this later to make sure
that none of our unlabeled data appears in the test set.
2) Select 1.01M TinyImages with keywords that don't appear in CIFAR-10. We use
these as training and validation data for the data sourcing model.
"""

import os
import argparse
import time

import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import logging

import numpy as np
from scipy.spatial.distance import cdist
import pickle
import pdb
import pandas as pd

from utils import load_cifar10_keywords
from utils import load_tinyimage_subset

cudnn.benchmark = True

# ----------------------------- CONFIGURATION ----------------------------------
parser = argparse.ArgumentParser(
    description='Compute distances from Tiny Images to CIFAR10 test set \
    and split train/test indices for TI vs. CIFAR10 model')

#Path config
parser.add_argument('--data_dir', default='data', type=str,
                    help='directory where datasets are located: \
                    contains cifar10_keywords_unique_v7.json and \
                    tinyimage data')
parser.add_argument('--output_dir', type=str,
                    help='directory to store results')
parser.add_argument('--features', type=str, default='raw',
                    help='can be "raw" for raw image data, or path'
                         ' (inside data_dir) of numpy file with features')

# Distance config
parser.add_argument('--metric', type=str, choices=['l2', 'cosine'],
                    default='l2',
                    help='Distance measure to use')
parser.add_argument('--subtract_mean', action='store_true', default=False,
                    help='Subtract feature mean before computing distance')

# General 
parser.add_argument('--batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for training')
parser.add_argument('--print_freq', '-s', default=5, type=int, metavar='N',
                    help='how many batches between log writes')
parser.add_argument('--num_workers', type=int, default=80,
                    help='Number of workers for data loading')


# parse args, etc.
args = parser.parse_args()

# define tensor conversion
def to_tensor(x, data_dtype):
    if data_dtype == 'uint8':
        t = torch.Tensor(x) / 255
    else:
        t = torch.Tensor(x)
    if args.subtract_mean:
        t -= t.mean(dim=-1, keepdim=True)
    return t

class MemmapDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, item):
        return to_tensor(self.data[item],
                         self.data.dtype)

    def __len__(self):
        return len(self.data)

# define distance computation
class NNDistance(nn.Module):
    def __init__(self, metric, target_set):
        super().__init__()
        self.metric = metric

        if metric == 'l2':
            target_set = target_set.t()
            self.register_buffer('target_set_norm_squared',
                                 (target_set ** 2).sum(dim=0, keepdim=True))
        elif metric == 'cosine':
            target_set = target_set / target_set.norm(dim=-1, keepdim=True)
            target_set = target_set.t()

        self.register_buffer('target_set', target_set)

    def forward(self, x):
        if self.metric == 'l2':
            # quadratic expansion is much faster because everything else wastes
            # memory
            dists = ((x ** 2).sum(dim=-1, keepdim=True) +
                     self.target_set_norm_squared -
                     2 * torch.mm(x, self.target_set)).clamp_min_(0.).sqrt_()

        elif self.metric == 'cosine':
            x = x / x.norm(dim=-1, keepdim=True)
            dists = 1.0 - torch.mm(x, self.target_set)
        else:
            raise ValueError('Unknown distance metric %s' % self.metric)

        return torch.min(dists, dim=1)


def main():
    batch_size = args.batch_size

    # Create dataset and loader
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'distance.log')),
            logging.StreamHandler()
        ])

    logging.info('Distance to CIFAR-10 test set')
    logging.info('Args: %s', args)

    num_images = 79302017
    if args.features == 'raw':
        data_path = os.path.join(args.data_dir, 'tiny_images.bin')
        data_dtype = 'uint8'
    else:
        data_path = os.path.join(args.data_dir, args.features)
        data_dtype = 'float32'

    out_path = os.path.join(args.output_dir, 'distance_to_cifar10_test.pickle')
    if os.path.exists(out_path):
        logging.info('Distance file exists, skipping computation')
    else: 
        # loading data 
        data = np.memmap(data_path, mode='r', dtype=data_dtype,
                         order='C').reshape(num_images, -1)
        data_loader = DataLoader(MemmapDataset(data),
                                 shuffle=False, batch_size=batch_size,
                                 num_workers=args.num_workers, pin_memory=True)
    
        # get cifar10 test indices for target set
        logging.info('Getting CIFAR10 test set images')
        keyword_data = load_cifar10_keywords(args.data_dir)
        cifar10_test_indices = np.array([x['nn_index'] for x in keyword_data[-10000:]])
        cifar10_test_data = to_tensor(data[cifar10_test_indices], data.dtype)
    
        logging.info('Initializing NN computation')
        nn_distance = DataParallel(NNDistance(args.metric, cifar10_test_data)).cuda()
        
        logging.info('Running distance computation')
        output_batches = []
        num_batches = len(data_loader)
        count = prev_count = 0
        start_time = time.time()
        for i, batch in enumerate(data_loader):
            batch = batch.cuda()
            with torch.no_grad():
                output_batch = [x.cpu().numpy() for x in nn_distance(batch)]
            output_batches.append(output_batch)
            count += len(output_batch[0])

            # Logging
            if (i + 1) % args.print_freq == 0 or i == num_batches - 1:
                increment = count - prev_count
                elapsed = time.time() - start_time
                logging.info('Processed %d/%d images (%.4g%%), %.3g images/sec' %
                             (count, num_images, 100 * count / num_images,
                              increment / elapsed))
                start_time = time.time()
                prev_count = count
            
        # save the results
        out_path = os.path.join(args.output_dir, 'distance_to_cifar10_test.pickle')
        logging.info('Saving results to %s' % out_path)
        nn_distances, nn_indices = [np.concatenate(x) for x in zip(*output_batches)]
        nn_indices = nn_indices.astype('uint16')
        with open(out_path, 'wb') as f:
            pickle.dump(dict(nn_distances=nn_distances, nn_indices=nn_indices), f)
        logging.info('Saved results to %s' % out_path)

    
    #------------ Choosing indices for TI vs. Cifar10 task -------------#
    indices_path = os.path.join(args.output_dir, 'ti_vs_cifar_inds.pickle')
    if os.path.exists(indices_path):
        logging.info('TI indices for selection model file exists')
    else:
        # load TI indices of keywords found in CIFAR10
        tinyimage_indices, tinyimage_data = load_tinyimage_subset(args.data_dir)

        # put TI metadata in more convenient form
        tinyimage_metadata = pd.DataFrame()
        for key, val in tinyimage_indices.items():
            df = pd.DataFrame(val)
            df['keyword'] = key
            tinyimage_metadata = tinyimage_metadata.append(df, ignore_index=True)
        
        is_valid = np.ones(num_images, dtype=bool)
        is_valid[tinyimage_metadata.tinyimage_index.values] = 0
        valid_indices = np.where(is_valid)[0]
        num_valid = len(valid_indices)

        # choose indices at random
        np.random.seed(13)
        num_ti_train = int(1e6)
        num_ti_test = int(1e4)
        ti_indices = valid_indices[np.random.permutation(num_valid)[:(num_ti_train+num_ti_test)]]
        # save indices
        indices_path = os.path.join(args.output_dir, 'ti_vs_cifar_inds.pickle')
        with open(indices_path, 'wb') as f:
            pickle.dump(dict(train=ti_indices[:num_ti_train],
                             test=ti_indices[num_ti_train:]), f)
        logging.info('Saved train vs. test indices to %s' % indices_path)
        
if __name__ == '__main__':
    main()
