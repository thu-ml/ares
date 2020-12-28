''' The bit depth reduction defense method. '''

import tensorflow as tf
import numpy as np

from ares.defense.input_transformation import input_transformation


def bit_depth_reduce(xs, x_min, x_max, step_num, alpha=1e6):
    ''' Run bit depth reduce on xs.

    :param xs: A batch of images to apply bit depth reduction.
    :param x_min: The minimum value of xs.
    :param x_max: The maximum value of xs.
    :param step_num: Step number for bit depth reduction.
    :param alpha: Alpha for bit depth reduction.
    :return: Bit depth reduced xs.
    '''
    # due to tf.custom_gradient's limitation, we need a wrapper here
    @tf.custom_gradient
    def bit_depth_reduce_op(xs_tf):
        steps = x_min + np.arange(1, step_num, dtype=np.float32) / (step_num / (x_max - x_min))
        steps = steps.reshape([1, 1, 1, 1, step_num-1])
        tf_steps = tf.constant(steps, dtype=tf.float32)

        inputs = tf.expand_dims(xs_tf, 4)
        quantized_inputs = x_min + tf.reduce_sum(tf.sigmoid(alpha * (inputs - tf_steps)), axis=4)
        quantized_inputs = quantized_inputs / ((step_num-1) / (x_max - x_min))

        def bit_depth_reduce_grad(d_output):
            return d_output

        return quantized_inputs, bit_depth_reduce_grad

    return bit_depth_reduce_op(xs)


def bit_depth_reduction(step_num, alpha=1e6):
    ''' A decorator to add bit depth reduce input transformation to a Classifier or a ClassifierWithLogits.

    It would leave the original classifier's logits' gradient untouched, so that white box attacks could still be
    applied to the new classifier.

    :param step_num: Step number for bit depth reduction.
    :param alpha: Alpha for bit depth reduction.
    '''
    def args_fn(model):
        return (model.x_min, model.x_max, step_num, alpha)

    def kwargs_fn(_):
        return {}

    return lambda rs_class: input_transformation(rs_class, bit_depth_reduce, args_fn, kwargs_fn)
