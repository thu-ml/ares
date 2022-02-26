import json
import os

import torch
from PIL import Image

class ImageDataset_Adv(torch.utils.data.Dataset):

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        self.transform = transform
        self._indices = []
        for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet_test_image_ids.txt')):
            img_path = line.strip()
            class_map = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet_class_to_id_map.json')))
            class_ids, _ = img_path.split('/')
            self._indices.append((img_path, class_map[class_ids]))

    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, index):
        img_path, label = self._indices[index]
        img = Image.open(os.path.join(self.data_dir, img_path)).convert('RGB')
        label = int(label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label