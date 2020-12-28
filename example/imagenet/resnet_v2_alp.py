import sys
import os

THIRD_PARTY_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party'))
if THIRD_PARTY_PATH not in sys.path:
    sys.path.append(THIRD_PARTY_PATH)
MODULE_PATH = os.path.join(THIRD_PARTY_PATH, 'models/research/slim')
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

import tensorflow as tf
import numpy as np

import models.research.slim.nets.resnet_v2 as resnet_v2

from realsafe import ClassifierWithLogits
from realsafe.utils import get_res_path, download_res

slim = tf.contrib.slim

MODEL_PATH = get_res_path('./imagenet/imagenet64_alp025_2018_06_26.ckpt')


def load(session):
    model = ResnetV2ALP()
    model.load(session, MODEL_PATH)
    return model


def download(model_path):
    url = 'http://download.tensorflow.org/models/adversarial_logit_pairing/imagenet64_alp025_2018_06_26.ckpt.tar.gz'
    if not os.path.exists(model_path + '.meta'):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import tarfile
        download_res(url, model_path + '.tar.gz')
        t = tarfile.open(model_path + '.tar.gz')
        t.extractall(os.path.dirname(model_path))
        os.remove(model_path + '.tar.gz')


class ResnetV2ALP(ClassifierWithLogits):
    def __init__(self):
        ClassifierWithLogits.__init__(self, 1001, 0.0, 1.0, (64, 64, 3), tf.float32, tf.int32)

    def _logits_and_labels(self, xs_ph):
        xs_ph = xs_ph * 2.0 - 1.0
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_50(xs_ph, self.n_class, is_training=False, reuse=tf.AUTO_REUSE)
        predicts = tf.nn.softmax(logits)
        predicted_labels = tf.cast(tf.argmax(predicts, 1), tf.int32)
        return logits, predicted_labels

    def load(self, session, model_path):
        x_input = tf.placeholder(tf.float32, shape=(None, *self.x_shape))
        with tf.contrib.framework.arg_scope(resnet_v2.resnet_arg_scope()):
            _, _ = resnet_v2.resnet_v2_50(x_input, self.n_class, is_training=False, reuse=tf.AUTO_REUSE)
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v2_50'))
        saver.restore(session, model_path)


if __name__ == '__main__':
    download(MODEL_PATH)
