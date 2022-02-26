from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

import torch
import torch.nn.functional as F
from torch.autograd import Variable as V
from models.wrn import WideResNet
import torchvision.transforms as transforms


def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)

def flip_labels_C(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)

    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C


class CIFAR10(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root='~/home-nfs/dan/cifar_data', train=True, gold=True, gold_fraction=0.1,
                 corruption_prob=0, corruption_type='unif', transform=None, target_transform=None,
                 download=False, shuffle_indices=None, distinguish_gold=True, seed=1):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.gold = gold
        self.gold_fraction = gold_fraction
        self.corruption_prob = corruption_prob

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            self.train_coarse_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                    num_classes = 10
                else:
                    self.train_labels += entry['fine_labels']
                    self.train_coarse_labels += entry['coarse_labels']
                    num_classes = 100
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

            if gold is True:
                if shuffle_indices is None:
                    indices = np.arange(50000)
                    shuffled_train_labels = self.train_labels
                    if self.gold_fraction >= 0.05:
                        while len(set(shuffled_train_labels[:int(gold_fraction * 50000)])) < num_classes:
                            np.random.shuffle(indices)
                            shuffled_train_labels = list(np.array(self.train_labels)[indices])
                    else:
                        gold_indices = []
                        for c in range(num_classes):
                            gold_indices.extend(indices[np.asarray(self.train_labels) == c][:int(self.gold_fraction * 50000 / num_classes)])
                        indices = np.array(gold_indices + list(set(range(50000)) - set(gold_indices)))
                else:
                    indices = shuffle_indices

                self.train_data = self.train_data[indices][:int(gold_fraction * 50000)]
                if distinguish_gold:
                    # this ad-hoc move is done so we can identify which examples are
                    # gold/trusted and which are silver/unstrusted
                    self.train_labels = list(np.array(self.train_labels)[indices][:int(gold_fraction * 50000)] + num_classes)
                else:
                    self.train_labels = list(np.array(self.train_labels)[indices][:int(gold_fraction * 50000)])
                self.shuffle_indices = indices
            else:
                indices = np.arange(len(self.train_data)) if shuffle_indices is None else shuffle_indices
                self.train_data = self.train_data[indices][int(gold_fraction * 50000):]
                self.train_labels = list(np.array(self.train_labels)[indices][int(gold_fraction * 50000):])
                if corruption_type == 'hierarchical':
                    self.train_coarse_labels = list(np.array(self.train_coarse_labels)[indices][int(gold_fraction * 50000):])

                if corruption_type == 'unif':
                    C = uniform_mix_C(self.corruption_prob, num_classes)
                elif corruption_type == 'flip':
                    C = flip_labels_C(self.corruption_prob, num_classes, seed=seed)
                elif corruption_type == 'hierarchical':
                    assert num_classes == 100, 'You must use CIFAR-100 with the hierarchical corruption.'
                    coarse_fine = []
                    for i in range(20):
                        coarse_fine.append(set())
                    for i in range(len(self.train_labels)):
                        coarse_fine[self.train_coarse_labels[i]].add(self.train_labels[i])
                    for i in range(20):
                        coarse_fine[i] = list(coarse_fine[i])

                    C = np.eye(num_classes) * (1 - corruption_prob)

                    for i in range(20):
                        tmp = np.copy(coarse_fine[i])
                        for j in range(len(tmp)):
                            tmp2 = np.delete(np.copy(tmp), j)
                            C[tmp[j], tmp2] += corruption_prob * 1/len(tmp2)
                elif corruption_type == 'clabels':
                    net = WideResNet(40, num_classes, 2, dropRate=0.3).cuda()
                    model_name = './cifar{}_labeler'.format(num_classes)
                    net.load_state_dict(torch.load(model_name))
                    net.eval()
                else:
                    assert False, "Invalid corruption type '{}' given. Must be in {'unif', 'flip', 'hierarchical'}".format(corruption_type)

                np.random.seed(seed)
                if corruption_type == 'clabels':
                    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
                    std = [x / 255 for x in [63.0, 62.1, 66.7]]

                    test_transform = transforms.Compose(
                        [transforms.ToTensor(), transforms.Normalize(mean, std)])

                    # obtain sampling probabilities
                    sampling_probs = []
                    print('Starting labeling')

                    for i in range((len(self.train_labels) // 64) + 1):
                        current = self.train_data[i*64:(i+1)*64]
                        current = [Image.fromarray(current[i]) for i in range(len(current))]
                        current = torch.cat([test_transform(current[i]).unsqueeze(0) for i in range(len(current))], dim=0)

                        data = V(current).cuda()
                        logits = net(data)
                        smax = F.softmax(logits / 5)  # temperature of 1
                        sampling_probs.append(smax.data.cpu().numpy())


                    sampling_probs = np.concatenate(sampling_probs, 0)
                    print('Finished labeling 1')

                    new_labeling_correct = 0
                    argmax_labeling_correct = 0
                    for i in range(len(self.train_labels)):
                        old_label = self.train_labels[i]
                        new_label = np.random.choice(num_classes, p=sampling_probs[i])
                        self.train_labels[i] = new_label
                        if old_label == new_label:
                            new_labeling_correct += 1
                        if old_label == np.argmax(sampling_probs[i]):
                            argmax_labeling_correct += 1
                    print('Finished labeling 2')
                    print('New labeling accuracy:', new_labeling_correct / len(self.train_labels))
                    print('Argmax labeling accuracy:', argmax_labeling_correct / len(self.train_labels))
                else:    
                    for i in range(len(self.train_labels)):
                        self.train_labels[i] = np.random.choice(num_classes, p=C[self.train_labels[i]])
                    self.corruption_matrix = C

        else:
            f = self.test_list[0][0]
            file = os.path.join(root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            if self.gold is True:
                return int(self.gold_fraction * 50000)
            else:
                return 50000 - int(self.gold_fraction * 50000)
        else:
            return 10000

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)


class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
