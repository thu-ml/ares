import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

# class Imagenet(Dataset):
#     def __init__(self,txt_file_path,image_dir,data_start=0,data_end=-1,transform=None,clip=True):
#         # self.csv_file=pd.read_csv(csv_file_path)
#         self.txt_file_path = txt_file_path
#         self.image_dir=image_dir
#         self.transform=transform
#         # self.image_size = image_size
#         self.data_start = data_start
#         self.data_end = data_end
#         self.clip = clip
#         with open(self.txt_file_path) as f:
#             self.lines = f.readlines()[self.data_start:self.data_end]

#     def __len__(self):
#         # return len(os.listdir(self.image_dir))
#         return len(self.lines)

#     def __getitem__(self,idx):
#         if torch.is_tensor(idx):
#             idx=idx.tolist()

#         line = self.lines[idx]
#         image_name, label = line.replace('\n', '').split(' ')
#         image_path=os.path.join(self.image_dir,image_name)
#         image=np.array(Image.open(image_path).convert('RGB'))
#         if self.clip:
#             height, width = image.shape[0], image.shape[1]  # pylint: disable=E1136  # pylint/issues/3139
#             center = int(0.875 * min(int(height), int(width)))
#             offset_height, offset_width = (height - center + 1) // 2, (width - center + 1) // 2
#             image = image[offset_height:offset_height+center, offset_width:offset_width+center, :]
#         image = Image.fromarray(image) #.resize(self.image_size)
#         if self.transform:
#             image=self.transform(image)

#         return image, int(label)


class Imagenet(Dataset):
    def __init__(self,txt_file_path,txt_target_path,image_dir,data_start=0,data_end=-1,transform=None,clip=True):
        # self.csv_file=pd.read_csv(csv_file_path)
        self.txt_file_path = txt_file_path
        self.txt_target_path = txt_target_path
        self.image_dir=image_dir
        self.transform=transform
        # self.image_size = image_size
        self.data_start = data_start
        self.data_end = data_end
        self.clip = clip
        with open(self.txt_file_path) as f:
            self.lines = f.readlines()[self.data_start:self.data_end]
        with open(self.txt_target_path) as f:
            self.lines1 = f.readlines()[self.data_start:self.data_end]

    def __len__(self):
        # return len(os.listdir(self.image_dir))
        return len(self.lines)  #, len(self.lines1)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()

        line = self.lines[idx]
        line1 = self.lines1[idx]
        image_name, label = line.replace('\n', '').split(' ')
        image_name, target_label = line1.replace('\n', '').split(' ')
        image_path=os.path.join(self.image_dir,image_name)
        image=np.array(Image.open(image_path).convert('RGB'))
        if self.clip:
            height, width = image.shape[0], image.shape[1]  # pylint: disable=E1136  # pylint/issues/3139
            center = int(0.875 * min(int(height), int(width)))
            offset_height, offset_width = (height - center + 1) // 2, (width - center + 1) // 2
            image = image[offset_height:offset_height+center, offset_width:offset_width+center, :]
        image = Image.fromarray(image) #.resize(self.image_size)
        if self.transform:
            image=self.transform(image)

        return image, int(label), int(target_label)