import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
from ares.utils.registry import registry
from ares.utils.model import NormalizeByChannelMeanStd

ngf = 64

class GeneratorResnet(nn.Module):
    def __init__(self, inception=False, data_dim='high'):
        '''
        :param inception (bool): if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim (str): for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        '''
        super(GeneratorResnet, self).__init__()
        self.inception = inception
        self.data_dim = data_dim
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        if self.data_dim == 'high':
            self.resblock3 = ResidualBlock(ngf * 4)
            self.resblock4 = ResidualBlock(ngf * 4)
            self.resblock5 = ResidualBlock(ngf * 4)
            self.resblock6 = ResidualBlock(ngf * 4)
            # self.resblock7 = ResidualBlock(ngf*4)
            # self.resblock8 = ResidualBlock(ngf*4)
            # self.resblock9 = ResidualBlock(ngf*4)

        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input):
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        if self.data_dim == 'high':
            x = self.resblock3(x)
            x = self.resblock4(x)
            x = self.resblock5(x)
            x = self.resblock6(x)
            # x = self.resblock7(x)
            # x = self.resblock8(x)
            # x = self.resblock9(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)
        return (torch.tanh(x) + 1) / 2 # Output range [0 1]


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual


def get_gaussian_kernel(kernel_size=3, pad=2, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, padding=kernel_size-pad,  bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

def load_netG(net_name='inception_v3', ckpt_netG='', mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    '''The function to load netG.'''
    if net_name=='inception_v3':
        netG=GeneratorResnet(inception=True)
    else:
        netG=GeneratorResnet()

    netG.load_state_dict(torch.load(ckpt_netG))
    normalize = NormalizeByChannelMeanStd(mean, std)
    netG = nn.Sequential([netG, normalize])

    return netG

@registry.register_attack('cda')
class CDA(object):
    ''' Learning transferable adversarial perturbations. A trasnfer-based black-box attack.

    Example:
        >>> from ares.utils.registry import registry
        >>> from ares.attack import load_netG
        >>> model = load_netG()
        >>> attacker_cls = registry.get_attack('cda')
        >>> attacker = attacker_cls(model)
        >>> adv_images = attacker(images, labels, target_labels)

    - Supported distance metric: 1, 2, np.inf.
    - References: https://openreview.net/forum?id=sIDvIyR5I1R.
    '''
    def __init__(self, netG, device='cuda', norm=np.inf, eps=4/255, gk=True):
        '''The initialize function for PGD.

        Args:
            netG (torch.nn.Module): The proxy model to be attacked.
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            norm (float): The norm of distance calculation for adversarial constraint. Defaults to np.inf.
            eps (float): The maximum perturbation range epsilon.
            gk (bool): Whether to use gaussian kernel during attack.
        '''
        self.eps = eps
        self.p = norm
        assert self.p==np.inf, 'The supported epsilon is np.inf.'
        self.device = device
        self.netG = netG.to(self.device)
        self.gk=gk
    
    def __call__(self, images=None, labels=None, target_labels=None):
        '''This function perform attack on target images with corresponding labels 
        and target labels for target attack.

        Args:
            images (torch.Tensor): The images to be attacked. The images should be torch.Tensor with shape [N, C, H, W] and range [0, 1].
            labels (torch.Tensor): The corresponding labels of the images. The labels should be torch.Tensor with shape [N, ]
            target_labels (torch.Tensor): The target labels for target attack. The labels should be torch.Tensor with shape [N, ]

        Returns:
            torch.Tensor: Adversarial images with value range [0,1].

        '''
        batchsize = images.shape[0]
        images, labels = images.to(self.device), labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        # advimage = images.clone().detach().requires_grad_(True).to(self.device)

        #initialize kernel
        if self.gk:
            kernel_size=3
            pad=2
            sigma=1
            kernel=get_gaussian_kernel(kernel_size=kernel_size, pad=pad, sigma=sigma).to(self.device)

        # get adv image
        adv=self.netG(images).detach()
        if self.gk:
            adv=kernel(adv)
        if not images.shape[2]==adv.shape[2]:
            resized_adv = F.interpolate(input=adv, size=images.shape[2])
        else:
            resized_adv = adv
        adv=torch.min(torch.max(resized_adv, images-self.eps), images+self.eps)
        adv=torch.clamp(adv, 0.0, 1.0)
        
        return adv
