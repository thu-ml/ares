'''
This file provides a wrapper class for InceptionV3 model
(https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py) for ImageNet dataset.
'''

import sys
import os
THIRD_PARTY_PATH = '../../third_party/models/research/slim'
THIRD_PARTY_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), THIRD_PARTY_PATH))
sys.path.append(THIRD_PARTY_PATH)


import tensorflow as tf
import urllib

from nets import inception_v3

from realsafe import ClassifierWithLogits
from realsafe.model.loader import get_res_path

slim = tf.contrib.slim


def load(session):
    model = InceptionV3()
    model.load(session)
    return model


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

    def load(self, session):
        x_input = tf.placeholder(self.x_dtype, shape=(None,) + self.x_shape)
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            inception_v3.inception_v3(x_input, num_classes=self.n_class, is_training=False, reuse=tf.AUTO_REUSE)

        model_path = get_res_path('./imagenet/inception_v3.ckpt')
        if not os.path.exists(model_path):
            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
            urllib.request.urlretrieve('http://ml.cs.tsinghua.edu.cn/~yinpeng/downloads/inception_v3.ckpt', model_path)

        saver = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        saver.restore(session, model_path)
