''' This file provides a wrapper class for Fast_AT (https://github.com/locuslab/fast_adversarial) model for ImageNet dataset. '''

import sys
import os

import torch
import torchvision.models as models
import tensorflow as tf

from ares.model.pytorch_wrapper import pytorch_classifier_with_logits
from ares.utils import get_res_path

MODEL_PATH = get_res_path('./imagenet/imagenet_model_weights_4px.pth')


def load(_):
    model = Fast_AT()
    model.load()
    return model


@pytorch_classifier_with_logits(n_class=1000, x_min=0.0, x_max=1.0,
                                x_shape=(288, 288, 3), x_dtype=tf.float32, y_dtype=tf.int32)
class Fast_AT(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.model = models.__dict__['resnet50']()
        self.model = torch.nn.DataParallel(self.model).cuda()
        self._mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
        self._std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()

    def forward(self, x):
        x = x.transpose(1, 2).transpose(1, 3).contiguous()
        input_var = (x.cuda() - self._mean_torch) / self._std_torch
        labels = self.model(input_var)

        return labels.cpu()

    def load(self):
        checkpoint = torch.load(MODEL_PATH)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        if not os.path.exists(os.path.dirname(MODEL_PATH)):
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = 'https://drive.google.com/file/d/1UrNEtLWs-fjlM2GPb1JpBGtpffDuHH_4/view'
        print('Please download "{}" to "{}".'.format(url, MODEL_PATH))
