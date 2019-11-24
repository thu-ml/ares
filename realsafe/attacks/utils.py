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
