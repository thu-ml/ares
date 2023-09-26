import os
import gdown
import torch
from collections import OrderedDict
from ares.model.wideresnet import create_wres28_10, create_wres34_10, create_wres34_10_fn
from ares.model.preact_resnet import create_preact_res18
from ares.model.cifar_model_zoo import cifar_model_zoo
from ares.utils.model import NormalizeByChannelMeanStd
from ares.utils.registry import registry

def filter_state_dict(state_dict):
    '''The function to filter state dict'''
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    elif 'net' in state_dict.keys():
        state_dict = state_dict['net']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module.' in k:
            k = k[7:]
        if 'basic_net.' in k:
            k = k[10:]
        new_state_dict[k] = v
    return new_state_dict

@registry.register_model('CifarCLS')
class CifarCLS(torch.nn.Module):
    '''The class to create cifar10 model.'''
    def __init__(self, model_name, normalize=True):
        '''
        Args:
            model_name (str): The model name in the cifar10 model zoo.
            normalize (bool): Whether interating the normalization layer into the model.
        '''
        super().__init__()
        self.model_name = model_name
        self.normalize = normalize
        self.arch = cifar_model_zoo[self.model_name]['model']
        mean=cifar_model_zoo[self.model_name]['mean']
        std=cifar_model_zoo[self.model_name]['std']

        if self.arch == 'preact_resnet18':
            self.model = create_preact_res18()
        elif self.arch == 'wresnet34_10_fn':
            self.model = create_wres34_10_fn()
        elif self.arch == 'wresnet28_10':
            self.model = create_wres28_10()
        elif self.arch == 'wresnet34_10':
            self.model = create_wres34_10()
        else:
            raise ValueError('Model not supported.')
        self.url = cifar_model_zoo[self.model_name]['url']
        self.pt_name = cifar_model_zoo[self.model_name]['pt']
        self.model_path = os.path.join(registry.get_path('cache_dir'), self.pt_name)
        gdown.download(self.url, self.model_path, quiet=False, resume=True)
        self.load()
        
        if self.normalize:
            normalization = NormalizeByChannelMeanStd(mean=mean, std=std)
            self.model = torch.nn.Sequential(normalization, self.model)


    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): The input images. The images should be torch.Tensor with shape [N, C, H, W] and range [0, 1].

        Returns:
            torch.Tensor: The output logits with shape [N D].

        '''

        labels = self.model(x)
        return labels

    def load(self):
        '''The function to load ckpt.'''
        checkpoint = torch.load(self.model_path, map_location='cpu')
        checkpoint = filter_state_dict(checkpoint)

        self.model.load_state_dict(checkpoint)
        self.model.eval()