from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import math
from pytorch_ares.dataset_torch.imagenet_dataset import Imagenet

def interpolation_(interpolation_mode):
    if interpolation_mode == 'bicubic':
        inter = 3
    elif interpolation_mode == 'nearest':
        inter = 0
    elif interpolation_mode == 'bilinear':
        inter = 2
    elif interpolation_mode == 'box':
        inter = 4
    elif interpolation_mode == 'hamming':
        inter = 5
    elif interpolation_mode == 'lanczos':
        inter = 1
    return inter



def datasets(data_name, batchsize,inputsize, crop_pct,interpolation1, cifar10_path, imagenet_val_path, imagenet_targrt_path,imagenet_path): 
    if data_name == "imagenet":
        print('Files already downloaded and verified')
        size = int(math.floor(inputsize / crop_pct)) 
        test_transform = transforms.Compose([transforms.Resize(size, interpolation=interpolation_(interpolation1)), transforms.CenterCrop(inputsize), 
             transforms.ToTensor()]) 
        # print('test transforms: ', test_transform) 
       
        imagenet = Imagenet(imagenet_val_path,imagenet_targrt_path, imagenet_path,
                                data_start=0,data_end=-1,transform=test_transform, clip=True )
        test_loader = DataLoader(imagenet, batch_size= batchsize, shuffle=False, num_workers= 4, pin_memory= False, drop_last= False)
       
        test_loader.name = "imagenet"
        test_loader.batch = batchsize
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        cifar = CIFAR10(root=cifar10_path, train=False, download=True, transform=transform)
        test_loader = DataLoader(cifar, batch_size=batchsize, shuffle=False, num_workers=1, pin_memory= False, drop_last= False)
        test_loader.name = "cifar10"
        test_loader.batch = batchsize
    return test_loader