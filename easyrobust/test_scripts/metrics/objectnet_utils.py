import os
import json
from torchvision import datasets
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

mapping_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imagenet_pytorch_id_to_objectnet_id.json')
with open(mapping_file,"r") as f:
    objectnet_mapping = json.load(f)
    objectnet_mapping = {int(k): v for k, v in objectnet_mapping.items()}

def imageNetIDToObjectNetID(prediction_class):
    for i in range(len(prediction_class)):
        if prediction_class[i] in objectnet_mapping:
            prediction_class[i] = objectnet_mapping[prediction_class[i]]
        else:
            prediction_class[i] = -1

# a dataset define which returns img_path
class ObjectNetDataset(datasets.DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ObjectNetDataset, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = []
        label_set = set()
        for k, v in objectnet_mapping.items():
            label_set.add(v)

        # filter samples non-overlap with imagenet
        for img_path, label in self.samples:
            if label in label_set:
                self.imgs.append((img_path, label))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        path, target = self.imgs[index]
        sample = self.loader(path)
        width, height = sample.size
        cropArea = (2, 2, width-2, height-2)
        sample = sample.crop(cropArea)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
