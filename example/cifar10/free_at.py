'''
This file provides a wrapper class for Free_AT (https://github.com/ashafahi/free_adv_train) model
for CIFAR-10 dataset.
'''

import sys
import os

THIRD_PARTY_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party'))
if THIRD_PARTY_PATH not in sys.path:
    sys.path.append(THIRD_PARTY_PATH)

from functools import partial
import tensorflow as tf

from ares.model import ClassifierWithLogits
from ares.utils import get_res_path

from cifar10_challenge.model import Model

MODEL_PATH = get_res_path('./cifar10/free_at/c10seed1cifar10_m8_eps8.0_b128')


def load(session):
    model = Free_AT()
    model.load(session, MODEL_PATH)
    return model


def convert_model(model_path):
    session = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    reader = tf.train.load_checkpoint(os.path.join(model_path, 'checkpoint-79999'))
    var_list = list(reader.get_variable_to_dtype_map().keys())
    for var in var_list:
        tf.Variable(reader.get_tensor(var), name=var.replace(var, 'free_at/'+var))
    session.run(tf.global_variables_initializer())

    import shutil
    shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)

    saver = tf.train.Saver()
    saver.save(session, os.path.join(model_path, 'free_at.ckpt'))


class Free_AT(ClassifierWithLogits):
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
        with tf.variable_scope("free_at"):
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
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(session, os.path.join(model_path, 'free_at.ckpt'))


if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        if not os.path.exists(os.path.dirname(MODEL_PATH)):
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = 'https://drive.google.com/file/d/16Wv8z7lI__I_NH6yXWw2cCkc68t_67lS/view'
        print('Please download "{}" and unzip it to "{}", and then run this command again to convert the model.'.format(url, MODEL_PATH))
    else:
        convert_model(MODEL_PATH)
