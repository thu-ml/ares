import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['resnet152_fd', ]


def tf_pad(x, kernel_size=3, stride=2):
    """For stride = 2 or stride = 3"""
    if x.shape[2] % stride == 0:
        padH = max(kernel_size - stride, 0)
    else:
        padH = max(kernel_size - (x.shape[2] % stride), 0)
    if x.shape[3] % stride == 0:
        padW = max(kernel_size - stride, 0)
    else:
        padW = max(kernel_size - (x.shape[3] % stride), 0)
    pad_top = padH // 2
    pad_bottom = padH - pad_top
    pad_left = padW // 2
    pad_right = padW - pad_left
    x_padded = torch.nn.functional.pad(
        x, pad=(pad_left, pad_right, pad_top, pad_bottom))
    return x_padded


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class NonLocal(nn.Module):

    def __init__(self, channels, inplace=False, embed=True, softmax=True):
        super().__init__()
        self.channels = channels
        self.inplace = inplace
        self.embed = embed
        self.softmax = softmax
        
        if self.embed:
            self.embedding_theta = nn.Conv2d(channels, channels//2, kernel_size=1, stride=1, bias=False)
            self.embedding_phi   = nn.Conv2d(channels, channels//2, kernel_size=1, stride=1, bias=False)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        n, c, h, w = x.shape
        res = x

        if self.embed:
            theta = self.embedding_theta(x)
            phi   = self.embedding_phi(x)
            g = x
        else:
            theta, phi, g = x, x, x

        if c > h * w or self.softmax:
            f = torch.einsum('niab,nicd->nabcd', theta, phi)
            if self.softmax:
                orig_shape = f.shape
                f = f.reshape(-1, h * w, h * w)
                f = f / torch.sqrt(torch.tensor(c, device=f.device, dtype=f.dtype))
                f = F.softmax(f, dim=-1)
                f = f.reshape(orig_shape)
            f = torch.einsum('nabcd,nicd->niab', f, g)
        else:
            f = torch.einsum('nihw,njhw->nij', phi, g)
            f = torch.einsum('nij,nihw->njhw', f, theta)
            
        if not self.softmax:
            f = f / torch.tensor(h * w, device=f.device, dtype=f.dtype)
        f = f.reshape(x.shape)
        f = self.conv(f)       
        f = self.bn(f)
        return f + res 

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes,)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.GELU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = tf_pad(out, kernel_size=3, stride=self.stride)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetDenoiseModel(nn.Module):
    def __init__(self, block, layers, num_classes=1000, denoise=True, **kwargs):
        super(ResNetDenoiseModel, self).__init__()
        self.inplanes = 64
        self.denoise = denoise
        
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.GELU()

        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.group0 = self._make_layer(block, 64,  layers[0], stride=1)
        self.group1 = self._make_layer(block, 128, layers[1], stride=2)
        self.group2 = self._make_layer(block, 256, layers[2], stride=2)
        self.group3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        if self.denoise:
            layers.append(NonLocal(planes * 4))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x[:, [2,1,0], :, :]
        
        x = tf_pad(x, kernel_size=7, stride=2)
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = tf_pad(x, kernel_size=3, stride=2)
        x = self.pool0(x)
        
        x = self.group0(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x 

def _resnet(arch, block, layers, **kwargs):
    model = ResNetDenoiseModel(block, layers, **kwargs)
    
    return model

def resnet152_fd(**kwargs):
    '''The function to create resnet152 model of feature denoising.'''
    kwargs['denoise'] = True
    return _resnet('resnet152_fd', Bottleneck, [3, 8, 36, 3], **kwargs)
