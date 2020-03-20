'''
This file provides a wrapper class for fd (https://github.com/facebookresearch/ImageNet-Adversarial-Training) model for
ImageNet dataset.
'''

import sys
import os
THIRD_PARTY_PATH = '../../third_party/ImageNet-Adversarial-Training'
THIRD_PARTY_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), THIRD_PARTY_PATH))
sys.path.append(THIRD_PARTY_PATH)

import tensorflow as tf
import nets

ResNetDenoiseModel = nets.ResNetDenoiseModel

from tensorpack import TowerContext
from tensorpack.tfutils import get_model_loader

from realsafe import ClassifierWithLogits
from realsafe.utils import get_res_path

import argparse


def load(session):
    model = ResNet152_Denoising()
    model.load(session)
    return model


class ResNet152_Denoising(ClassifierWithLogits):
    def __init__(self):
        ClassifierWithLogits.__init__(self, 1000, 0.0, 1.0, (224, 224, 3), tf.float32, tf.int32)
        args = argparse.Namespace()
        args.depth = 152
        self.model = ResNetDenoiseModel(args)

    def _logits_and_labels(self, xs_ph):
        xs_ph = xs_ph * 2.0 - 1.0
        xs_ph = xs_ph[:, :, :, ::-1]
        xs_ph = tf.transpose(xs_ph, [0, 3, 1, 2])
        with TowerContext('', is_training=False):
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                logits = self.model.get_logits(xs_ph)
        predicts = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(predicts, 1)
        return logits, predicted_labels

    def load(self, session):
        x_input = tf.placeholder(tf.float32, shape=(None,) + self.x_shape)
        x_input = tf.transpose(x_input, [0, 3, 1, 2])
        with TowerContext('', is_training=False):
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                logits = self.model.get_logits(x_input)

        model_path = get_res_path('./imagenet/R152-Denoise.npz')
        url = 'http://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/realsafe/feature_denosing/R152-Denoise.npz'
        if not os.path.exists(model_path):
            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))

            from six.moves import urllib
            urllib.request.urlretrieve(url, model_path)

        get_model_loader(model_path).init(session)
