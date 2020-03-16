import sys
import os
THIRD_PARTY_PATH = '../../third_party/models/research/slim'
THIRD_PARTY_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), THIRD_PARTY_PATH))
sys.path.append(THIRD_PARTY_PATH)

import tensorflow as tf
import numpy as np

import urllib

from nets import resnet_v2

from realsafe import ClassifierWithLogits
from realsafe.model.loader import get_res_path

slim = tf.contrib.slim


def load(session):
    model = ResnetV2ALP()
    model.load(session)
    return model


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

    def load(self, session):
        x_input = tf.placeholder(tf.float32, shape=(None, *self.x_shape))
        with tf.contrib.framework.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_50(x_input, self.n_class, is_training=False, reuse=tf.AUTO_REUSE)
        model_path = get_res_path('./imagenet/alp')
        url = 'http://download.tensorflow.org/models/adversarial_logit_pairing/imagenet64_alp025_2018_06_26.ckpt.tar.gz'
        fname = os.path.join(model_path, url.split('/')[-1])
        if not os.path.exists(fname):
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            urllib.request.urlretrieve(url, fname)

            import tarfile
            t = tarfile.open(fname)
            t.extractall(model_path)
            print('Extracted model')

        saver = tf.train.Saver()
        saver.restore(session, fname.split('.tar.gz')[0])
