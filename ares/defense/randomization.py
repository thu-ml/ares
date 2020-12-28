''' The randomization defense method, which applies random . '''

import tensorflow as tf

from ares.defense.input_transformation import input_transformation


def randomize(xs, scale_min=0.875, pad_value=0.0):
    ''' Apply random rescaling and padding to xs.

    :param xs: A batch of inputs for some classifier.
    :param scale_min: The random rescaling rate would be chosen between ``scale_min`` and 1.0.
    :param pad_value: ``constant_values`` parameter for the ``tf.pad`` method.
    :return: A new tensor with same shape and dtype as xs.
    '''
    ratio = tf.random.uniform((), minval=scale_min, maxval=1.0)
    height, width = tf.cast(xs.shape[1].value * ratio, tf.int32), tf.cast(xs.shape[2].value * ratio, tf.int32)
    xs_rescaled = tf.image.resize(xs, (height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                  align_corners=True, preserve_aspect_ratio=False)
    height_rem, width_rem = xs.shape[1].value - height, xs.shape[2].value - width
    pad_left = tf.random_uniform((), 0, width_rem, dtype=tf.int32)
    pad_right = width_rem - pad_left
    pad_top = tf.random_uniform((), 0, height_rem, dtype=tf.int32)
    pad_bottom = height_rem - pad_top
    xs_padded = tf.pad(xs_rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                       constant_values=pad_value)
    xs_padded.set_shape(xs.shape)
    return xs_padded


def randomization(scale_min=0.875, pad_value=0.0):
    ''' A decorator to apply randomize rescaling and padding to input of the classifier.

    :param scale_min: The random rescaling rate would be chosen between ``scale_min`` and 1.0.
    :param pad_value: ``constant_values`` parameter for the ``tf.pad`` method.
    '''
    def args_fn(_):
        return (scale_min, pad_value)

    def kwargs_fn(_):
        return {}

    return lambda rs_class: input_transformation(rs_class, randomize, args_fn, kwargs_fn)
