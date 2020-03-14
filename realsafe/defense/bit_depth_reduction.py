''' The bit depth reduction defense method. '''

import tensorflow as tf
import numpy as np

from realsafe.model import Classifier, ClassifierWithLogits


def bit_depth_reduce(xs, x_min, x_max, step_num, alpha=1e6):
    '''
    Run bit depth reduce on xs.
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
    '''
    A decorator to add bit depth reduce input transformation to a Classifier or a ClassifierWithLogits. It would leave
    the original classifier's logits' gradient untouched, so that white box attacks could still be applied to the new
    classifier.
    :param step_num: Step number for bit depth reduction.
    :param alpha: Alpha for bit depth reduction.
    '''
    def decorator(rs_class):
        if issubclass(rs_class, ClassifierWithLogits):  # prefer using ClassifierWithLogits
            class Wrapper(rs_class):  # directly inherit the classifier's class
                def _logits_and_labels(self, xs):  # implement ClassifierWithLogits' interface
                    xs_bdr = bit_depth_reduce(xs, self.x_min, self.x_max, step_num, alpha)
                    # we need to call the _logits_and_labels() instead of logits_and_labels() here
                    return super()._logits_and_labels(xs_bdr)
            return Wrapper
        elif issubclass(rs_class, Classifier):
            class Wrapper(rs_class):  # directly inherit the classifier's class
                def _labels(self, xs):  # implement Classifier's interface
                    xs_bdr = bit_depth_reduce(xs, self.x_min, self.x_max, step_num, alpha)
                    # we need to call the _logits() instead of logits() here
                    return super()._labels(xs_bdr)
            return Wrapper
        else:
            raise TypeError('bit_depth_reduction() requires a Classifier or a ClassifierWithLogits class.')

    return decorator
