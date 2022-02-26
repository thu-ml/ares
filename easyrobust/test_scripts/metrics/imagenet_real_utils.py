import os
import json
from torchvision import datasets
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

real_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real.json')
with open(real_json) as real_labels:
    real_labels = json.load(real_labels)
    real_labels = {f'ILSVRC2012_val_{i + 1:08d}.JPEG': labels for i, labels in enumerate(real_labels)}


# a dataset define which returns img_path
class ImageFolderReturnsPath(datasets.DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolderReturnsPath, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, path
