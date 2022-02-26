'''

Style Augmentation: Data Augmentation via Style Randomization (CVPR2019 workshops)
Paper link: https://arxiv.org/abs/1809.05375

Modified by Xiaofeng Mao 
2021.8.30
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import numpy as np
import sys
from os.path import join, dirname

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StylePredictor(nn.Module):
    def __init__(self):
        super(StylePredictor, self).__init__()
        # InceptionV3, truncated at Mixed_6e:
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        # this layer predicts the style embedding from mean-pooled inception features:
        self.fc = nn.Linear(768,100)

    def forward(self, x):
        # x: B x 3 x H x W in [0,1]

        x = x.clone()

        # PyTorch InceptionV3 requires two preprocessing steps; one which is common to all PyTorch ImageNet models, and one which goes from that common preprocess to Google's Inception preproc, which is different: https://github.com/pytorch/vision/pull/101 
        
        # PyTorch preproc (https://github.com/pytorch/examples/blob/master/imagenet/main.py):
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.255
        # Google preproc (https://github.com/pytorch/vision/blob/c1746a252372dc62a73766a5772466690fc1b8a6/torchvision/models/inception.py#L72-L76):
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5


        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        
        # mean-pool the convolutional features:
        x = x.mean(dim=3).mean(dim=2) # B x 768

        # predict style embedding:
        x = self.fc(x)

        return x



class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class ConvInRelu(nn.Module):
    def __init__(self,channels_in,channels_out,kernel_size,stride=1):
        super(ConvInRelu,self).__init__()
        self.n_params = 0
        self.channels = channels_out
        self.reflection_pad = nn.ReflectionPad2d(int(np.floor(kernel_size/2)))
        self.conv = nn.Conv2d(channels_in,channels_out,kernel_size,stride,padding=0)
        self.instancenorm = nn.InstanceNorm2d(channels_out)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        # x: B x C_in x H x W

        x = self.reflection_pad(x)
        x = self.conv(x) # B x C_out x H x W
        x = self.instancenorm(x) # B x C_out x H x W
        x = self.relu(x) # B x C_out x H x W
        return x


class UpsampleConvInRelu(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, upsample, stride=1, activation=nn.ReLU):
        super(UpsampleConvInRelu, self).__init__()
        self.n_params = channels_out * 2
        self.upsample = upsample
        self.channels = channels_out

        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride)
        self.instancenorm = nn.InstanceNorm2d(channels_out)
        self.fc_beta = nn.Linear(100,channels_out)
        self.fc_gamma = nn.Linear(100,channels_out)
        if activation:
            self.activation = activation(inplace=False)
        else:
            self.activation = None
        
    def forward(self, x, style):
        # x: B x C_in x H x W
        # style: B x 100

        beta = self.fc_beta(style).unsqueeze(2).unsqueeze(3) # B x C_out x 1 x 1
        gamma = self.fc_gamma(style).unsqueeze(2).unsqueeze(3) # B x C_out x 1 x 1

        if self.upsample:
            x = self.upsample_layer(x)
        x = self.reflection_pad(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        x = gamma * x
        x += beta
        if self.activation:
            x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    # modelled after that used by Johnson et al. (2016)
    # see https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.n_params = channels * 4
        self.channels = channels

        self.reflection_pad = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels,channels,3,stride=1,padding=0)
        self.instancenorm = nn.InstanceNorm2d(channels)
        self.fc_beta1 = nn.Linear(100,channels)
        self.fc_gamma1 = nn.Linear(100,channels)
        self.fc_beta2 = nn.Linear(100,channels)
        self.fc_gamma2 = nn.Linear(100,channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels,channels,3,stride=1,padding=0)
        
    def forward(self, x, style):
        # x: B x C x H x W  
        # style: B x self.n_params
        
        beta1 = self.fc_beta1(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
        gamma1 = self.fc_gamma1(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
        beta2 = self.fc_beta2(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
        gamma2 = self.fc_gamma2(style).unsqueeze(2).unsqueeze(3) # B x C x 1 x 1

        y = self.reflection_pad(x)
        y = self.conv1(y)
        y = self.instancenorm(y)
        y = gamma1 * y
        y += beta1
        y = self.relu(y)
        y = self.reflection_pad(y)
        y = self.conv2(y)
        y = self.instancenorm(y)
        y = gamma2 * y
        y += beta2
        return x + y


class Ghiasi(nn.Module):
    def __init__(self):
        super(Ghiasi,self).__init__()
        self.layers = nn.ModuleList([
            ConvInRelu(3,32,9,stride=1),
            ConvInRelu(32,64,3,stride=2),
            ConvInRelu(64,128,3,stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            UpsampleConvInRelu(128,64,3,upsample=2),
            UpsampleConvInRelu(64,32,3,upsample=2),
            UpsampleConvInRelu(32,3,9,upsample=None,activation=None)
        ])

        self.n_params = sum([layer.n_params for layer in self.layers])
    
    def forward(self,x,styles):
        # x: B x 3 x H x W
        # styles: B x 100 batch of style embeddings
        
        for i, layer in enumerate(self.layers):
            if i < 3:
                # first three layers do not perform renormalization (see style_normalization_activations in the original source: https://github.com/tensorflow/magenta/blob/master/magenta/models/arbitrary_image_stylization/nza_model.py)
                x = layer(x)
            else:
                x = layer(x, styles)
        
        return torch.sigmoid(x)

class StyleAugmentor(nn.Module):
    def __init__(self):
        super(StyleAugmentor,self).__init__()

        # create transformer and style predictor networks:
        self.ghiasi = Ghiasi()
        self.stylePredictor = StylePredictor()
        self.ghiasi.to(device)
        self.stylePredictor.to(device)

        # load checkpoints:
        checkpoint_ghiasi = model_zoo.load_url('http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/pretrained_models/checkpoint_transformer.pth')
        checkpoint_stylepredictor = model_zoo.load_url('http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/pretrained_models/checkpoint_stylepredictor.pth')
        checkpoint_embeddings = model_zoo.load_url('http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/pretrained_models/checkpoint_embeddings.pth')
        
        # load weights for ghiasi and stylePredictor, and mean / covariance for the embedding distribution:
        self.ghiasi.load_state_dict(checkpoint_ghiasi['state_dict_ghiasi'],strict=False)
        self.stylePredictor.load_state_dict(checkpoint_stylepredictor['state_dict_stylepredictor'],strict=False)

        # load mean imagenet embedding:
        self.imagenet_embedding = checkpoint_embeddings['imagenet_embedding_mean'] # mean style embedding for ImageNet
        self.imagenet_embedding = self.imagenet_embedding.to(device)

        # get mean and covariance of PBN style embeddings:
        self.mean = checkpoint_embeddings['pbn_embedding_mean']
        self.mean = self.mean.to(device) # 1 x 100
        self.cov = checkpoint_embeddings['pbn_embedding_covariance']
        
        # compute SVD of covariance matrix:
        u, s, vh = np.linalg.svd(self.cov.numpy())
        
        self.A = np.matmul(u,np.diag(s**0.5))
        self.A = torch.tensor(self.A).float().to(device) # 100 x 100
        # self.cov = cov(Ax), x ~ N(0,1)
    
    def sample_embedding(self,n):
        # n: number of embeddings to sample
        # returns n x 100 embedding tensor
        embedding = torch.randn(n,100).to(device) # n x 100
        embedding = torch.mm(embedding,self.A.transpose(1,0)) + self.mean # n x 100
        return embedding

    def forward(self,x,alpha=0.5,downsamples=0,embedding=None,useStylePredictor=True):
        # augments a batch of images with style randomization
        # x: B x C x H x W image tensor
        # alpha: float in [0,1], controls interpolation between random style and original style
        # downsamples: int, number of times to downsample by factor of 2 before applying style transfer
        # embedding: B x 100 tensor, or None. Use this embedding if provided.
        # useStylePredictor: bool. If True, we use the inception based style predictor to compute the original style embedding for the input image, and use that for interpolation. If False, we use the mean ImageNet embedding instead, which is slightly faster.

        # style embedding for when alpha=0:
        base = self.stylePredictor(x) if useStylePredictor else self.imagenet_embedding

        if downsamples:
            assert(x.size(2) % 2**downsamples == 0)
            assert(x.size(3) % 2**downsamples == 0)
            for i in range(downsamples):
                x = nn.functional.avg_pool2d(x,2)

        if embedding is None:
            # sample a random embedding
            embedding = self.sample_embedding(x.size(0))
        # interpolate style embeddings:
        embedding = alpha*embedding + (1-alpha)*base
        
        restyled = self.ghiasi(x,embedding)

        if downsamples:
            restyled = nn.functional.upsample(restyled,scale_factor=2**downsamples,mode='bilinear')
        
        return restyled.detach() # detach prevents the user from accidentally backpropagating errors into stylePredictor or ghiasi while training a downstream model

# class Image_Style_Transfer_Augment(object):
#     def __init__(self):
#         self.augmentor = StyleAugmentor()

#     def __call__(self, im_torch):
#         im_torch = im_torch.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
#         im_restyled = self.augmentor(im_torch)
#         return im_restyled
