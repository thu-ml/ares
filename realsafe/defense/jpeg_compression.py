''' The jpeg compression defense method. '''

import tensorflow as tf

from realsafe.model import Classifier, ClassifierWithLogits


def jpeg_compress(xs, x_min, x_max, quality=95):
    '''
    Run jpeg compress on xs.
    :param xs: A batch of images to compress.
    :param x_min: The minimum value of xs.
    :param x_max: The maximum value of xs.
    :param quality: Jpeg compress quality.
    :return: Compressed images tensor with same numerical scale to the input image.
    '''
    # due to tf.custom_gradient's limitation, we need a wrapper here
    @tf.custom_gradient
    def jpeg_compress_op(xs_tf):
        # batch_size x width x height x channel
        imgs = tf.cast((xs_tf - x_min) / ((x_max - x_min) / 255.0), tf.uint8)
        imgs_jpeg = tf.map_fn(lambda img: tf.image.decode_jpeg(tf.image.encode_jpeg(img, quality=quality)), imgs)
        imgs_jpeg.set_shape(xs_tf.shape)

        def jpeg_compress_grad(d_output):
            return d_output

        return tf.cast(imgs_jpeg, xs_tf.dtype) / (255.0 / (x_max - x_min)) + x_min, jpeg_compress_grad

    return jpeg_compress_op(xs)


def jpeg_compression(quality=95):
    '''
    A decorator to add jpeg compress input transformation to a Classifier or a ClassifierWithLogits. It would leave
    the original classifier's logits' gradient untouched, so that white box attacks could still be applied to the new
    classifier.
    :param quality: The jpeg compression quality.
    '''
    def decorator(rs_class):
        if issubclass(rs_class, ClassifierWithLogits):  # prefer using ClassifierWithLogits
            class Wrapper(rs_class):  # directly inherit the classifier's class
                def _logits_and_labels(self, xs):  # implement ClassifierWithLogits' interface
                    xs_jpeg = jpeg_compress(xs, self.x_min, self.x_max, quality)
                    # we need to call the _logits_and_labels() instead of logits_and_labels() here
                    return super()._logits_and_labels(xs_jpeg)
            return Wrapper
        elif issubclass(rs_class, Classifier):
            class Wrapper(rs_class):  # directly inherit the classifier's class
                def _labels(self, xs):  # implement Classifier's interface
                    xs_jpeg = jpeg_compress(xs, self.x_min, self.x_max, quality)
                    # we need to call the _logits() instead of logits() here
                    return super()._labels(xs_jpeg)
            return Wrapper
        else:
            raise TypeError('jpeg_compression() requires a Classifier or a ClassifierWithLogits class.')

    return decorator
