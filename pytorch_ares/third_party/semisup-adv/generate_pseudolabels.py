"""
Code for running a generating pseudolabels for unlabeled TinyImages data
"""
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import logging
import os
import pickle

import argparse

import numpy as np

from torchvision import transforms

import torch
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

import time

import pdb

parser = argparse.ArgumentParser(
    description='Apply standard trained model to generate labels on unlabeled data')
parser.add_argument('--model', '-m', default='wrn-28-10', type=str,
                    help='name of the model')
parser.add_argument('--model_dir', type=str,
                    help='path of checkpoint to standard trained model')
parser.add_argument('--model_epoch', '-e', default=200, type=int,
                    help='Number of epochs trained')
parser.add_argument('--data_dir', default='data/', type=str,
                    help='directory that has unlabeled data')
parser.add_argument('--data_filename', default='ti_top_50000_pred_v3.1.pickle', type=str,
                    help='name of the file with unlabeled data')
parser.add_argument('--output_dir', default='data/', type=str,
                    help='directory to save output')
parser.add_argument('--output_filename', default='pseudolabeled-top50k.pickle', type=str,
                    help='file to save output')

args = parser.parse_args()
if not os.path.exists(args.model_dir):
    raise ValueError('Model dir %s not found' % args.model_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.model_dir, 'prediction.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('Prediction on unlabeled data')
logging.info('Args: %s', args)


# Loading unlabeled data
with open(os.path.join(args.data_dir, args.data_filename), 'rb') as f:
    data = pickle.load(f)

# Loading model
checkpoint = torch.load(os.path.join(args.model_dir, 'checkpoint-epoch%d.pt' % args.model_epoch))
num_classes = checkpoint.get('num_classes', 10)
normalize_input = checkpoint.get('normalize_input', False)
model = get_model(args.model, 
                  num_classes=num_classes,
                  normalize_input=normalize_input)
model = nn.DataParallel(model).cuda()
model.load_state_dict(checkpoint['state_dict'])

model.eval()

unlabeled_data = CIFAR10('data', train=False, transform=ToTensor())
unlabeled_data.data = data['data']
# targets from the CIFAR10 vs. TI model
unlabeled_data.targets = list(data['extrapolated_targets'])
data_loader = torch.utils.data.DataLoader(unlabeled_data,
                                          batch_size=1000,
                                          num_workers=100,
                                          pin_memory=True)

# Running model on unlabeled data
predictions = []
for i, (batch, _) in enumerate(data_loader):
    _, preds = torch.max(model(batch.cuda()), dim=1)
    predictions.append(preds.cpu().numpy())

    if (i+1) % 10 == 0:
        print('Done %d/%d' % (i+1, len(data_loader)))

new_extrapolated_targets = np.concatenate(predictions)

new_targets = dict(extrapolated_targets=new_extrapolated_targets,
                   prediction_model=args.model_dir,
                   prediction_model_epoch=args.model_epoch)

out_path = os.path.join(args.output_dir, args.output_filename)
assert(not os.path.exists(out_path))
with open(out_path, 'wb') as f:
        pickle.dump(new_targets, f)
