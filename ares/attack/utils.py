import math

import tensorflow as tf
import numpy as np


def get_xs_ph(model, batch_size):
    return tf.placeholder(model.x_dtype, (batch_size, *model.x_shape))


def get_ys_ph(model, batch_size):
    return tf.placeholder(model.y_dtype, (batch_size,))


def maybe_to_array(x, target_len):
    arr = np.array(x)
    if arr.shape == ():
        return np.repeat(x, target_len)
    else:
        return arr


def get_unit(xs_flatten):
    xs_norm = tf.maximum(1e-12, tf.norm(xs_flatten, axis=1))
    xs_unit = xs_flatten / tf.expand_dims(xs_norm, 1)
    return xs_unit


def uniform_l_inf_noise(batch_size, d, eps, dtype):
    eps = tf.expand_dims(eps, 1)
    noise_unit = tf.random.uniform((batch_size, d), minval=-1, maxval=1, dtype=dtype)
    noise_r = tf.random.uniform((batch_size, 1), minval=0, maxval=1, dtype=dtype)
    return eps * noise_r * noise_unit


def uniform_l_2_noise(batch_size, d, eps, dtype):
    eps = tf.expand_dims(eps, 1)
    noise_unit = get_unit(tf.random.uniform((batch_size, d), minval=-1, maxval=1, dtype=dtype))
    noise_r = tf.random.uniform((batch_size, 1), minval=0, maxval=1, dtype=dtype)
    return eps * noise_r * noise_unit


class ConfigVar(object):
    def __init__(self, shape, dtype):
        if shape is None:
            self.var = tf.Variable(0, dtype=dtype)
        else:
            self.var = tf.Variable(tf.zeros(shape, dtype=dtype))
        self.ph = tf.placeholder(dtype, shape=shape)
        self.assign = self.var.assign(self.ph)


class Expectation(object):
    def __init__(self, x, iteration):
        self._var = tf.Variable(tf.zeros_like(x))
        self.reset = self._var.assign(tf.zeros_like(x))
        self.update = self._var.assign_add(x)
        self.val = self._var / iteration


def clip_eta_batch(xs, eta, distance_metric):
    if distance_metric == 'l_2':
        rank = len(xs.shape) - 1
        return tf.clip_by_norm(xs, eta, axes=[i + 1 for i in range(rank)])
    elif distance_metric == 'l_inf':
        return tf.clip_by_value(xs, tf.negative(eta), eta)
    else:
        raise NotImplementedError


def clip_eta(x, eta, distance_metric):
    if distance_metric == 'l_2':
        return tf.clip_by_norm(x, eta)
    elif distance_metric == 'l_inf':
        return tf.clip_by_value(x, tf.negative(eta), eta)
    else:
        raise NotImplementedError


def image_resize(imgs, height, width):
    return tf.image.resize(imgs, (height, width), method=tf.image.ResizeMethod.BILINEAR,
                           align_corners=True, preserve_aspect_ratio=False)


def scale(x, dst_min, dst_max, src_min, src_max):
    k = (dst_max - dst_min) / (src_max - src_min)
    b = dst_min - k * src_min
    return k * x + b


def split_trunks(xs, n):
    N = len(xs)
    trunks = []
    trunk_size = N // n
    if N % n == 0:
        for rank in range(n):
            start = rank * trunk_size
            trunks.append(xs[start:start + trunk_size])
    else:
        for rank in range(N % n):
            start = rank * (trunk_size + 1)
            trunks.append(xs[start:start + trunk_size + 1])
        for rank in range(N % n, n):
            start = rank * trunk_size + (N % n)
            trunks.append(xs[start:start + trunk_size])
    return trunks
