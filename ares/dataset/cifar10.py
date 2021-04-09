''' CIFAR-10 dataset. '''

import tensorflow as tf
import numpy as np

from keras.datasets.cifar10 import load_data

from ares.utils import get_res_path

PATH_TARGET = get_res_path('cifar10/target.npy')


def load_dataset_for_classifier(classifier, offset=0, load_target=False, target_label=None):
    ''' Get an CIFAR-10 dataset in tf.data.Dataset format.
    
    The first element of the dataset is the index, the second one is the image tensor with shape of the classifier's
    ``x_shape`` in the classifier's ``x_dtype``, the third one is the label in the classifier's ``y_dtype``. If
    ``load_target`` is true, the target label would be returned as the fourth element of the dataset.

    :param offset: Ignore the first ``offset`` images.
    :param load_target: Whether to load the target label.
    :param target_label: If it is a integer, the returned dataset would only include data points with this label.
    :return: A ``tf.data.Dataset`` instance.
    '''
    label_dtype = classifier.y_dtype
    x_shape, x_dtype, x_min, x_max = classifier.x_shape, classifier.x_dtype, classifier.x_min, classifier.x_max
    dataset = load_dataset(offset=offset, label_dtype=label_dtype, load_target=load_target, target_label=target_label)

    def scale(*ts):
        ts = list(ts)
        ts[1] = tf.reshape(tf.cast(ts[1], x_dtype) * ((x_max - x_min) / 255.0) + x_min, x_shape)
        return tuple(ts)

    return dataset.map(scale, num_parallel_calls=1)


def load_dataset(offset=0, label_dtype=tf.int32, load_target=False, target_label=None):
    ''' Get a CIFAR-10 dataset in tf.data.Dataset format.

    The first element of the dataset is the index, the second one is the image tensor with shape of (32, 32, 3) in
    ``tf.uint8``. If ``load_target`` is true, the target label would be returned as the fourth element of the dataset.

    :param offset: Ignore the first ``offset`` images.
    :param label_dtype: Label's data type.
    :param load_target: Whether to load the target label.
    :param target_label: If it is a integer, the returned dataset would only include data points with this label.
    :return: A ``tf.data.Dataset`` instance.
    '''
    _, (xs_test, ys_test) = load_data()
    ys_test = ys_test.reshape(len(ys_test))
    ts = tf.data.Dataset.from_tensor_slices(np.load(PATH_TARGET)[offset:])
    xs, ys = tf.data.Dataset.from_tensor_slices(xs_test[offset:]), tf.data.Dataset.from_tensor_slices(ys_test[offset:])
    ys = ys.map(lambda y: tf.cast(y, label_dtype))
    ts = ts.map(lambda t: tf.cast(t, label_dtype))

    ids = tf.data.Dataset.range(offset, len(ys_test))

    if target_label is not None:
        if load_target:
            return tf.data.Dataset.zip((ids, xs, ys, ts)).filter(lambda *ps: tf.math.equal(ps[2], target_label))
        else:
            return tf.data.Dataset.zip((ids, xs, ys)).filter(lambda *ps: tf.math.equal(ps[2], target_label))
    if load_target:
        return tf.data.Dataset.zip((ids, xs, ys, ts))
    else:
        return tf.data.Dataset.zip((ids, xs, ys))


if __name__ == '__main__':
    import os
    from ares.utils import download_res

    _ = load_data()
    if not os.path.exists(PATH_TARGET):
        os.makedirs(os.path.dirname(PATH_TARGET), exist_ok=True)
        download_res('https://ml.cs.tsinghua.edu.cn/~qian/ares/target.npy', PATH_TARGET)
