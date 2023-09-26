import os
import torch
from PIL import Image
    
class ImageNetDataset(torch.utils.data.Dataset):
    '''The class to create ImageNet dataset.'''

    def __init__(self, data_dir, meta_file, transform=None):
        """The function to initialize ImageNet class.

        Args:
            data_dir (str): The path to the dataset.
            meta_file (str): The path to the file containing image directories and labels.
            transform (torchvision.transforms): The transform for input image.
        """

        self.data_dir = data_dir
        self.meta_file = meta_file
        self.transform = transform
        self._indices = []
        for line in open(os.path.join(os.path.dirname(__file__), meta_file), encoding="utf-8"):
            img_path, label, target_label = line.strip().split(' ')
            self._indices.append((os.path.join(self.data_dir, img_path), label, target_label))

    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, index):
        img_path, label, target_label = self._indices[index]
        img = Image.open(img_path).convert('RGB')
        label = int(label)
        target_label=int(target_label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, target_label
