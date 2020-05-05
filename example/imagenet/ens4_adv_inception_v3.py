'''
This file provides a wrapper class for step L.L. on ensemble of 4 models
(https://github.com/tensorflow/models/blob/master/research/adv_imagenet_models/README.md) for ImageNet dataset. It use
same graph definition as inception_v3_*.py, so they would override each other if using in same graph.
'''

import sys
import os

import tensorflow as tf

from inception_v3 import inception_v3

from realsafe import ClassifierWithLogits
from realsafe.utils import get_res_path, download_res

slim = tf.contrib.slim

MODEL_PATH = get_res_path('./imagenet/ens4_adv_inception_v3')


def load(session):
    model = Ens4AdvInceptionV3()
    model.load(session, MODEL_PATH)
    return model


def download(model_path):
    if not os.path.exists(model_path):
        import tarfile

        os.makedirs(model_path)
        download_res('http://download.tensorflow.org/models/ens4_adv_inception_v3_2017_08_18.tar.gz',
                     os.path.join(model_path, 'ens4_adv_inception_v3_2017_08_18.tar.gz'))

        tar = tarfile.open(os.path.join(model_path, 'ens4_adv_inception_v3_2017_08_18.tar.gz'))
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name, model_path)

        os.remove(os.path.join(model_path, 'ens4_adv_inception_v3_2017_08_18.tar.gz'))


class Ens4AdvInceptionV3(ClassifierWithLogits):
    def __init__(self):
        ClassifierWithLogits.__init__(self, 1001, 0.0, 1.0, (299, 299, 3), tf.float32, tf.int32)

    def _logits_and_labels(self, xs_ph):
        xs_ph = xs_ph * 2.0 - 1.0
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, end_points = inception_v3.inception_v3(xs_ph, num_classes=self.n_class,
                                                           is_training=False, reuse=tf.AUTO_REUSE)
            predicted_labels = tf.cast(tf.argmax(end_points['Predictions'], 1), tf.int32)
        return logits, predicted_labels

    def load(self, session, model_path):
        x_input = tf.placeholder(self.x_dtype, shape=(None,) + self.x_shape)
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            inception_v3.inception_v3(x_input, num_classes=self.n_class, is_training=False, reuse=tf.AUTO_REUSE)
        saver = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        saver.restore(session, os.path.join(model_path, 'ens4_adv_inception_v3.ckpt'))


if __name__ == '__main__':
    download(MODEL_PATH)
