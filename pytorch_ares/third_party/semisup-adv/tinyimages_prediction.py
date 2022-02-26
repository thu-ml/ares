"""
Select unlabeled data from TinyImages using the trained data sourcing model
"""

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import logging
import os
import pickle

from utils import get_model, load_cifar10_keywords

import argparse

import numpy as np

from torchvision import transforms

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset


import time

import pdb

parser = argparse.ArgumentParser(
    description='Apply pretrained network on entire Tiny Images dataset')
# Model details
parser.add_argument('--model_path', type=str,
                    help='path of checkpoint to pretrained model')
parser.add_argument('--model', '-m', default='wrn-28-10', type=str,
                    help='name of the model')
# Directory details 
parser.add_argument('--data_dir', default='data/', type=str,
                    help='directory where datasets are located')
parser.add_argument('--output_dir', type=str,
                    help='directory where unlabeled dataset is to be located')
parser.add_argument('--output_filename', type=str,
                    help='Name of the pickle file containing unlabeled data')

# Unlabeled data details
parser.add_argument('--num_unlabeled', type=int, default=50000,
                    help='Number of unlabeled images in each class')
parser.add_argument('--l2_cutoff', type=float, default=2000/255,
                    help='l2 distance cutoff to choose valid indices')

# Running details
parser.add_argument('--batch_size', type=int, default=1000, metavar='N',
                    help='batch_size')
parser.add_argument('--save_freq', '-s', default=5, type=int, metavar='N',
                    help='how many batches between savings of the data')
parser.add_argument('--num_workers', type=int, default=10,
                    help='Number of workers for data loading')
parser.add_argument('--save_prelogits', action='store_true', default=False,
                    help='Save all prelogits to a (big) file')

# parse args, etc.
args = parser.parse_args()
batch_size = args.batch_size

if not os.path.exists(args.model_path):
    raise ValueError('Model path %s not found' % args.model_path)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.output_dir, 'prediction.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('Prediction on Tiny Images')
logging.info('Args: %s', args)


# create dataset and loader
num_images = 79302017
data_path = os.path.join(args.data_dir, 'tiny_images.bin')
data = np.memmap(data_path, mode='r', dtype='uint8', order='C'
                 ).reshape(num_images, 3, 32, 32).transpose((0, 1, 3, 2))


def to_tensor(x):
    t = torch.Tensor(x) / 255
    t -= torch.Tensor([0.4914, 0.4822, 0.4465]).reshape(3, 1 ,1)
    t /= torch.Tensor([0.2470, 0.2435, 0.2616]).reshape(3, 1 ,1)
    return t

# load model
checkpoint = torch.load(args.model_path)
num_classes = checkpoint['config']['model_config']['n_classes']
model = get_model(args.model, num_classes=num_classes, normalize_input=False)
model = DataParallel(model).cuda()
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# make data loader
class MemmapDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, item):
        return to_tensor(self.data[item])
        # return to_tensor(self.data[cifar10_indices[item]])

    def __len__(self):
        return len(self.data)
        # return len(cifar10_indices)

data_loader = DataLoader(MemmapDataset(data),
                         shuffle=False, batch_size=batch_size,
                         num_workers=args.num_workers, pin_memory=True)


# create output variables (mapped directly to hard-drive to save memory)
def _get_mode(path):
    return 'r+' if os.path.exists(path) else 'w+'


# Saving logits to logits.bin in output_dir
logits_path = os.path.join(args.output_dir, 'logits.bin')
logits_file = np.memmap(logits_path,
                        dtype=np.float32, mode=_get_mode(logits_path),
                        shape=(num_images, num_classes))

if args.save_prelogits:
    num_features = model.feature_size
    prelogits_path = os.path.join(args.output_dir, 'prelogits.bin')
    prelogits_file = np.memmap(prelogits_path,
                               dtype=np.float32, mode=_get_mode(prelogits_path),
                               shape=(num_images, num_features))
    
# run predictions
output_batches = []
num_batches = len(data_loader)
count = 0
start_time = time.time()

for i, batch in enumerate(data_loader):
    batch = batch.cuda()
    with torch.no_grad():
        logits, prelogits = model.forward(x=batch, return_prelogit=True)
        output_batches.append((logits.cpu().numpy(),
                               prelogits.cpu().numpy()))
        
    # Saving data
    if (i + 1) % args.save_freq == 0 or i == num_batches - 1:
        logits_batch, prelogits_batch = [np.concatenate(s)
                                         for s in zip(*output_batches)]
        increment = logits_batch.shape[0]
        logits_file[count:(count + increment)] = logits_batch
        if args.save_prelogits:
            prelogits_file[count:(count + increment)] = prelogits_batch

        output_batches = []

        count += increment

        elapsed = time.time() - start_time

        logging.info('Processed %d/%d images (%.4g%%), %.3g images/sec' %
                     (count, num_images, 100 * count / num_images,
                      increment / elapsed))

        start_time = time.time()

# Processing the logits file
ti_logits = np.array(logits_file)

ti_probs = np.exp(ti_logits - ti_logits.max(axis=-1, keepdims=True))
ti_probs /= ti_probs.sum(axis=-1, keepdims=True)
ti_predicted_labels = np.argmax(ti_logits, axis=-1)
ti_prediction_probs = ti_probs[np.arange(len(ti_probs)), ti_predicted_labels]

# Creating list of valid images to be used
num_images = 79302017
is_valid = np.ones(num_images, dtype='bool')

with open(os.path.join(args.data_dir, 'distance_to_cifar10_test.pickle'), 'rb') as f:
    ldd = pickle.load(f)
    nn_distances, nn_indices = ldd['nn_distances'], ldd['nn_indices']

# Removing TI images close to CIFAR-10 test set 
is_valid &= (nn_distances > args.l2_cutoff)

# Creating label balanced set by choosing top predictions of each label
inds = np.zeros(0, dtype='int')    
labels = list(range(10))
for label in labels:
    mask = is_valid & (ti_predicted_labels == label)
    inds = np.concatenate((inds,
                           np.where(mask)[0][np.argsort(ti_prediction_probs[mask])[-args.num_unlabeled:]]))

data = np.array(data[inds])
targets = np.concatenate([args.num_unlabeled * [label] for label in labels]).astype('int')
out = dict(data=data, extrapolated_targets=targets, ti_index=inds)

print('Created %s (%d images from %d candidates)' % (args.output_filename, len(data), is_valid.sum()))
with open(os.path.join(args.output_dir, args.output_filename ), 'wb') as f:
    pickle.dump(out, f)
