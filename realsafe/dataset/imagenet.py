''' ImageNet dataset (ILSVRC 2012). '''

import os
import tensorflow as tf
import numpy as np
from PIL import Image

from realsafe.model.loader import get_res_path

PATH_IMGS = get_res_path('./imagenet/ILSVRC2012_img_val')
PATH_VAL_TXT = get_res_path('./imagenet/val.txt')
PATH_TARGET_TXT = get_res_path('./imagenet/target.txt')


def load_dataset_for_classifier(classifier, offset=0, load_target=False, target_label=None, clip=True):
    '''
    Get an ImageNet dataset in tf.data.Dataset format. The first element of the dataset is the filename, the second one
    is the image tensor with shape of the classifier's `x_shape` in the classifier's `x_dtype`, the third one is the
    label in the classifier's `y_dtype`. If `load_target` is true, the target label would be returned as the fourth
    element of the dataset.
    :param offset: Ignore the first `offset` images.
    :param load_target: Whether to load the target label.
    :param target_label: If it is a integer, the returned dataset would only include data points with this label.
    :param clip: If it is true, the images would be clipped towards center.
    :return: A `tf.data.Dataset` instance.
    '''
    height, width = classifier.x_shape[:2]
    label_dtype = classifier.y_dtype
    x_dtype, x_min, x_max = classifier.x_dtype, classifier.x_min, classifier.x_max
    dataset = load_dataset(height, width, offset=offset, label_dtype=label_dtype,
                           load_target=load_target, target_label=target_label, clip=clip)

    def scale(*ts):
        ts = list(ts)
        ts[1] = tf.cast(ts[1], x_dtype) * ((x_max - x_min) / 255.0) + x_min
        return tuple(ts)

    return dataset.map(scale, num_parallel_calls=8)


def load_dataset(height, width, offset=0, label_dtype=tf.int32, load_target=False, target_label=None, clip=True):
    '''
    Get an ImageNet dataset in tf.data.Dataset format. The first element of the dataset is the filename, the second one
    is the image tensor with shape of (height, width, 3) in `tf.uint8`, the third one is the label. If `load_target` is
    true, the target label would be returned as the fourth element of the dataset.
    :param height: The target height.
    :param width: The target width.
    :param offset: Ignore the first `offset` images.
    :param label_dtype: Label's data type.
    :param load_target: Whether to load the target label.
    :param target_label: If it is a integer, the returned dataset would only include data points with this label.
    :param clip: If it is true, the images would be clipped towards center.
    :return: A `tf.data.Dataset` instance.
    '''
    filenames, labels = _load_txt(PATH_VAL_TXT)
    filenames, labels = filenames[offset:], labels[offset:]
    targets = None if not load_target else _load_txt(PATH_TARGET_TXT)[1][offset:]

    if target_label is not None:
        filenames, labels, targets = _filter_by_label(target_label, filenames, labels, targets)

    ds_filename = tf.data.Dataset.from_tensor_slices(filenames)
    ds_label = tf.data.Dataset.from_tensor_slices(labels).map(lambda x: tf.cast(x, label_dtype))

    if load_target:
        ds_target = tf.data.Dataset.from_tensor_slices(targets).map(lambda x: tf.cast(x, label_dtype))

    def map_fn(filename):
        image = tf.py_function(lambda x: _load_image(x.numpy().decode(), height, width, clip), [filename], tf.uint8)
        image.set_shape((height, width, 3))
        return image

    ds_image = ds_filename.map(map_fn, num_parallel_calls=8)

    if load_target:
        return tf.data.Dataset.zip((ds_filename, ds_image, ds_label, ds_target))
    return tf.data.Dataset.zip((ds_filename, ds_image, ds_label))


def _load_image(filename, to_height, to_width, clip):
    ''' Load image into uint8 tensor from file. '''
    img = Image.open(os.path.join(PATH_IMGS, filename))
    if img.mode != 'RGB':
        img = img.convert(mode='RGB')

    if clip:
        img = np.array(img)
        height, width = img.shape[0], img.shape[1]  # pylint: disable=E1136  # pylint/issues/3139
        center = int(0.875 * min(height, width))
        offset_height, offset_width = (height - center + 1) // 2, (width - center + 1) // 2
        img = img[offset_height:offset_height+center, offset_width:offset_width+center, :]
        img = Image.fromarray(img)

    return tf.convert_to_tensor(np.array(img.resize((to_height, to_width))))


def _load_txt(txt_filename):
    ''' Load images' filename and label from `txt_filename`. '''
    filenames, labels = [], []
    with open(txt_filename) as txt:
        for line in txt:
            filename, label = line.strip('\n').split(' ')
            filenames.append(filename)
            labels.append(int(label))
    return filenames, labels


def _filter_by_label(target_label, filenames, labels, targets):
    ''' Filter out image not in the target_label. '''
    r_filenames, r_labels, r_targets = [], [], []
    if targets is None:
        for filename, label in zip(filenames, labels):
            if label == target_label:
                r_filenames.append(filename)
                r_labels.append(label)
        return r_filenames, r_labels, None
    else:
        for filename, label, target in zip(filenames, labels, targets):
            if label == target_label:
                r_filenames.append(filename)
                r_labels.append(label)
                r_targets.append(target)
        return r_filenames, r_labels, r_targets
