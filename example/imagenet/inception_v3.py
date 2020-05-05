'''
This file provides a wrapper class for InceptionV3 model
(https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) for ImageNet dataset.
'''

import sys
import os

THIRD_PARTY_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party'))
if THIRD_PARTY_PATH not in sys.path:
    sys.path.append(THIRD_PARTY_PATH)
MODULE_PATH = os.path.join(THIRD_PARTY_PATH, 'models/research/slim')
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

import tensorflow as tf

import models.research.slim.nets.inception_v3 as inception_v3

from realsafe import ClassifierWithLogits
from realsafe.utils import get_res_path, download_res

slim = tf.contrib.slim

MODEL_PATH = get_res_path('./imagenet/inception_v3.ckpt')


def load(session):
    model = InceptionV3()
    model.load(session, MODEL_PATH)
    return model


def download(model_path):
    if not os.path.exists(model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        download_res('http://ml.cs.tsinghua.edu.cn/~yinpeng/downloads/inception_v3.ckpt', model_path)


class InceptionV3(ClassifierWithLogits):
    def __init__(self):
        ClassifierWithLogits.__init__(self, 1001, 0.0, 1.0, (299, 299, 3), tf.float32, tf.int32)

    def _logits_and_labels(self, xs_ph):
        xs_ph = xs_ph * 2.0 - 1.0

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, end_points = inception_v3.inception_v3(
                xs_ph,
                num_classes=self.n_class,
                is_training=False,
                reuse=tf.AUTO_REUSE)

            predicted_labels = tf.argmax(end_points['Predictions'], 1)

        return logits, predicted_labels

    def load(self, session, model_path):
        x_input = tf.placeholder(self.x_dtype, shape=(None,) + self.x_shape)
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            inception_v3.inception_v3(x_input, num_classes=self.n_class, is_training=False, reuse=tf.AUTO_REUSE)
        saver = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        saver.restore(session, model_path)


if __name__ == '__main__':
    download(MODEL_PATH)
