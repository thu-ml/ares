'''
This file provides a wrapper class for PGD Adversarial Training (https://github.com/MadryLab/cifar10_challenge) model
for CIFAR-10 dataset.
'''

import sys
import os

THIRD_PARTY_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party'))
if THIRD_PARTY_PATH not in sys.path:
    sys.path.append(THIRD_PARTY_PATH)

from functools import partial
import tensorflow as tf

from realsafe.model import ClassifierWithLogits
from realsafe.utils import get_res_path, download_res

MODEL_PATH = get_res_path('./cifar10/pgd_at')

from cifar10_challenge.model import Model


def load(session):
    model = PgdAT()
    model.load(session, MODEL_PATH)
    return model


def download(model_path):
    if not os.path.exists(os.path.join(model_path, 'models/adv_trained/checkpoint')):
        zip_path = os.path.join(model_path, 'adv_trained.zip')
        if not os.path.exists(zip_path):
            if not os.path.exists(os.path.dirname(zip_path)):
                os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        download_res('https://www.dropbox.com/s/g4b6ntrp8zrudbz/adv_trained.zip?dl=1', zip_path)
        import zipfile
        zipfile.ZipFile(zip_path).extractall(model_path)
        os.remove(zip_path)


class PgdAT(ClassifierWithLogits):
    def __init__(self):
        super().__init__(10, 0.0, 255.0, (32, 32, 3), tf.float32, tf.int32)
        self.mode = 'eval'
        self._stride_arr = partial(Model._stride_arr, self)
        self._batch_norm = partial(Model._batch_norm, self)
        self._residual = partial(Model._residual, self)
        self._decay = partial(Model._decay, self)
        self._conv = partial(Model._conv, self)
        self._relu = partial(Model._relu, self)
        self._fully_connected = partial(Model._fully_connected, self)
        self._global_avg_pool = partial(Model._global_avg_pool, self)

    def _logits_and_labels(self, xs):
        return self._build_model(xs, True)

    def _build_model(self, xs, reuse):
        # based on the Model._build_model function
        with tf.variable_scope('input', reuse=reuse):
            input_standardized = tf.map_fn(tf.image.per_image_standardization, xs)
            x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1))

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        res_func = self._residual

        # wide residual network (https://arxiv.org/abs/1605.07146v1)
        # use filters = [16, 16, 32, 64] for a non-wide version
        filters = [16, 160, 320, 640]

        # Update hps.num_residual_units to 9
        with tf.variable_scope('unit_1_0', reuse=reuse):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]), activate_before_residual[0])
        for i in range(1, 5):
            with tf.variable_scope('unit_1_%d' % i, reuse=reuse):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        with tf.variable_scope('unit_2_0', reuse=reuse):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]), activate_before_residual[1])
        for i in range(1, 5):
            with tf.variable_scope('unit_2_%d' % i, reuse=reuse):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        with tf.variable_scope('unit_3_0', reuse=reuse):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]), activate_before_residual[2])
        for i in range(1, 5):
            with tf.variable_scope('unit_3_%d' % i, reuse=reuse):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        with tf.variable_scope('unit_last', reuse=reuse):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, 0.1)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logit', reuse=reuse):
            logits = self._fully_connected(x, 10)

        labels = tf.cast(tf.argmax(logits, 1), tf.int32)
        return logits, labels

    def load(self, session, model_path):
        var_list_pre = set(tf.global_variables())
        self._build_model(tf.placeholder(tf.float32, (None, *self.x_shape)), tf.AUTO_REUSE)
        var_list_post = set(tf.global_variables())
        var_list = list(var_list_post - var_list_pre)
        if len(var_list) > 0:
            checkpoint = tf.train.latest_checkpoint(os.path.join(model_path, 'models/adv_trained/'))
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(session, checkpoint)


if __name__ == '__main__':
    download(MODEL_PATH)
