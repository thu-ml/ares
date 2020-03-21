''' CIFAR-10 dataset. '''

import tensorflow as tf
from keras.datasets.cifar10 import load_data


def load_dataset_for_classifier(classifier, offset=0, target_label=None):
    '''
    Get an CIFAR-10 dataset in tf.data.Dataset format. The first element of the dataset is the index, the second one is
    the image tensor with shape of the classifier's `x_shape` in the classifier's `x_dtype`, the third one is the label
    in the classifier's `y_dtype`.
    :param offset: Ignore the first `offset` images.
    :param target_label: If it is a integer, the returned dataset would only include data points with this label.
    :return: A `tf.data.Dataset` instance.
    '''
    label_dtype = classifier.y_dtype
    x_shape, x_dtype, x_min, x_max = classifier.x_shape, classifier.x_dtype, classifier.x_min, classifier.x_max
    dataset = load_dataset(offset=offset, label_dtype=label_dtype, target_label=target_label)

    def scale(*ts):
        ts = list(ts)
        ts[1] = tf.reshape(tf.cast(ts[1], x_dtype) * ((x_max - x_min) / 255.0) + x_min, x_shape)
        return tuple(ts)

    return dataset.map(scale, num_parallel_calls=8)


def load_dataset(offset=0, label_dtype=tf.int32, target_label=None):
    '''
    Get a CIFAR-10 dataset in tf.data.Dataset format. The first element of the dataset is the index, the second one is
    the image tensor with shape of (32, 32, 3) in `tf.uint8`, the third one is the label.
    :param offset: Ignore the first `offset` images.
    :param label_dtype: Label's data type.
    :param target_label: If it is a integer, the returned dataset would only include data points with this label.
    :return: A `tf.data.Dataset` instance.
    '''
    _, (xs_test, ys_test) = load_data()
    ys_test = ys_test.reshape(len(ys_test))
    xs, ys = tf.data.Dataset.from_tensor_slices(xs_test[offset:]), tf.data.Dataset.from_tensor_slices(ys_test[offset:])
    ys = ys.map(lambda y: tf.cast(y, label_dtype))

    if target_label is not None:
        return tf.data.Dataset.zip((xs, ys)).filter(lambda x, y: tf.math.equal(y, target_label))
    return tf.data.Dataset.zip((xs, ys))


if __name__ == '__main__':
    _ = load_data()
