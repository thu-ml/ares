'''
This file provides a wrapper class for step L.L. on ensemble of 4 models
(https://github.com/tensorflow/models/blob/master/research/adv_imagenet_models/README.md) for ImageNet dataset.
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

        os.makedirs(model_path, exist_ok=True)
        download_res('http://download.tensorflow.org/models/ens4_adv_inception_v3_2017_08_18.tar.gz',
                     os.path.join(model_path, 'ens4_adv_inception_v3_2017_08_18.tar.gz'))

        tar = tarfile.open(os.path.join(model_path, 'ens4_adv_inception_v3_2017_08_18.tar.gz'))
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name, model_path)

        session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        reader = tf.train.load_checkpoint(os.path.join(model_path, 'ens4_adv_inception_v3.ckpt'))
        var_list = list(reader.get_variable_to_dtype_map().keys())
        for var in var_list:
            tf.Variable(reader.get_tensor(var), name=var.replace('InceptionV3', 'Ens4InceptionV3'))
        session.run(tf.global_variables_initializer())

        import shutil
        shutil.rmtree(model_path)
        os.makedirs(model_path, exist_ok=True)

        saver = tf.train.Saver()
        saver.save(session, os.path.join(model_path, 'ens4_adv_inception_v3.ckpt'))


class Ens4AdvInceptionV3(ClassifierWithLogits):
    def __init__(self):
        ClassifierWithLogits.__init__(self, 1001, 0.0, 1.0, (299, 299, 3), tf.float32, tf.int32)

    def _logits_and_labels(self, xs_ph):
        xs_ph = xs_ph * 2.0 - 1.0
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, end_points = inception_v3.inception_v3(xs_ph, num_classes=self.n_class,
                                                           is_training=False, reuse=tf.AUTO_REUSE,
                                                           scope='Ens4InceptionV3')
            predicted_labels = tf.cast(tf.argmax(end_points['Predictions'], 1), tf.int32)
        return logits, predicted_labels

    def load(self, session, model_path):
        x_input = tf.placeholder(self.x_dtype, shape=(None,) + self.x_shape)
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            inception_v3.inception_v3(x_input, num_classes=self.n_class, is_training=False, reuse=tf.AUTO_REUSE,
                                      scope='Ens4InceptionV3')
        saver = tf.train.Saver(slim.get_model_variables(scope='Ens4InceptionV3'))
        saver.restore(session, os.path.join(model_path, 'ens4_adv_inception_v3.ckpt'))


if __name__ == '__main__':
    download(MODEL_PATH)
