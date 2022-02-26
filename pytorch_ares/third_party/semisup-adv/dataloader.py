"""
Based on code from https://github.com/hysts/pytorch_shake_shake
"""

import numpy as np
import torch
import torchvision
import os
import pickle
from torch.utils import data
import pdb

def get_loader(batch_size, num_workers, use_gpu):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])

    dataset_dir = 'data'
    train_dataset = torchvision.datasets.CIFAR10(
        dataset_dir, train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        dataset_dir, train=False, transform=test_transform, download=True)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader


""" Function to load data for cifar0 vs TI classifier
"""
def get_cifar10_vs_ti_loader(batch_size, num_workers, use_gpu,
                             cifar_fraction=0.5, dataset_dir='data', 
                             logger=None):

    # Normalization values for CIFAR-10
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        dataset_dir, train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        dataset_dir, train=False, transform=test_transform, download=True)

    # Reading tinyimages and appropriate train/test indices
    logger.info('Reading tiny images')
    ti_path = os.path.join(dataset_dir, 'tiny_images.bin')
    ti_data = np.memmap(ti_path, mode='r', dtype='uint8', order='F',
                        shape=(32, 32, 3, 79302017)).transpose([3, 0, 1, 2])
    
    logger.info('Size of tiny images {}'.format(ti_data.size))
    ti_indices_path = os.path.join(dataset_dir,
                                   'ti_vs_cifar_inds.pickle')
    with open(ti_indices_path, 'rb') as f:
        ti_indices = pickle.load(f)
    logger.info('Loaded TI indices')
    
    for dataset, name in zip((train_dataset, test_dataset), ('train', 'test')):
        dataset.data = np.concatenate((dataset.data, ti_data[ti_indices[name]]))
        # All tinyimages are given label 10
        dataset.targets.extend([10] * len(ti_indices[name]))

    logger.info('Calling train sampler')
    # Balancing training batches with CIFAR10 and TI
    train_sampler = BalancedSampler(
        train_dataset.targets, batch_size,
        balanced_fraction=cifar_fraction,
        num_batches=int(50000 / (batch_size * cifar_fraction)),
        label_to_balance=10, 
        logger=logger)
    
    logger.info('Created train sampler')
    train_loader = data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=use_gpu,
    )

    logger.info('Created train loader')
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    logger.info('Created test loader')
    return train_loader, test_loader


class BalancedSampler(data.Sampler):
    def __init__(self, labels, batch_size,
                 balanced_fraction=0.5,
                 num_batches=None,
                 label_to_balance=-1, 
                 logger=None):
        logger.info('Inside balanced sampler')
        self.minority_inds = [i for (i, label) in enumerate(labels)
                              if label != label_to_balance]
        self.majority_inds = [i for (i, label) in enumerate(labels)
                              if label == label_to_balance]
        self.batch_size = batch_size
        balanced_batch_size = int(batch_size * balanced_fraction)
        self.minority_batch_size = batch_size - balanced_batch_size

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = int(
                np.ceil(len(self.minority_inds) / self.minority_batch_size))

        super().__init__(labels)

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            minority_inds_shuffled = [self.minority_inds[i]
                                      for i in
                                      torch.randperm(len(self.minority_inds))]
            # Cycling through permutation of minority indices
            for sup_k in range(0, len(self.minority_inds),
                               self.minority_batch_size):
                if batch_counter == self.num_batches:
                    break
                batch = minority_inds_shuffled[
                        sup_k:(sup_k + self.minority_batch_size)]
                # Appending with random majority indices
                if self.minority_batch_size < self.batch_size:
                    batch.extend(
                        [self.majority_inds[i] for i in
                         torch.randint(high=len(self.majority_inds),
                                       size=(self.batch_size - len(batch),),
                                       dtype=torch.int64)])
                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches
