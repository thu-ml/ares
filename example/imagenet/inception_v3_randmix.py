'''
This file provides a wrapper class for RandMix
(https://worksheets.codalab.org/worksheets/0x822ba2f9005f49f08755a84443c76456/) model for ImageNet dataset.
'''

import sys
import os

THIRD_PARTY_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party'))
if THIRD_PARTY_PATH not in sys.path:
    sys.path.append(THIRD_PARTY_PATH)
MODULE_PATH = os.path.join(THIRD_PARTY_PATH, 'models/research/slim')
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

import tensorflow as tf

import models.research.slim.nets.inception_v3 as inception_v3

from ares import ClassifierWithLogits
from ares.utils import get_res_path, download_res

slim = tf.contrib.slim

MODEL_PATH = get_res_path('./imagenet/inception_v3.ckpt')


def load(session):
    model = InceptionV3()
    model.load(session, MODEL_PATH)
    return model


class InceptionV3(ClassifierWithLogits):
    def __init__(self):
        ClassifierWithLogits.__init__(self, 1001, 0.0, 1.0, (299, 299, 3), tf.float32, tf.int32)
        self.n_clusters = 5
        self.noise_level = 32.0 / 255.0
        self.num_ensemble = 10

    def _logits_and_labels(self, xs_ph):
        xs_ph = xs_ph * 2.0 - 1.0
        batch_size = tf.shape(xs_ph)[0]
        xs_tile = tf.tile(tf.expand_dims(xs_ph, 1), [1, self.num_ensemble, 1, 1, 1])
        xs_tile = tf.reshape(xs_tile, (-1,) + self.x_shape)

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            with tf.variable_scope('RandDisc'):
                xs_tile = iterative_clustering_layer(source=xs_tile, n_clusters=self.n_clusters, sigma=10, alpha=10,
                                                     noise_level_1=self.noise_level, noise_level_2=self.noise_level)
            logits, _ = inception_v3.inception_v3(xs_tile, num_classes=self.n_class, is_training=False, reuse=True)
            logits = tf.reshape(logits, [batch_size, self.num_ensemble, -1])
            logits = tf.reduce_mean(logits, axis=1)
            labels = tf.cast(tf.argmax(logits, 1), tf.int32)

        return logits, labels

    def load(self, session, model_path):
        x_input = tf.placeholder(self.x_dtype, shape=(None,) + self.x_shape)
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            inception_v3.inception_v3(x_input, num_classes=self.n_class, is_training=False, reuse=tf.AUTO_REUSE)
        saver = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        saver.restore(session, model_path)


def iterative_clustering_layer(source, n_clusters, sigma, alpha, noise_level_1, noise_level_2):
    source1 = source + tf.random_normal(tf.shape(source)) * noise_level_1
    source2 = source + tf.random_normal(tf.shape(source)) * noise_level_2
    centroids = sample_centroid_with_kpp(source1, 100, n_clusters, sigma)
    image, _ = rgb_clustering(source2, centroids, alpha)
    return image


def sample_centroid_with_kpp(images, n_samples, n_clusters, sigma):
    _, width, height, channel = images.get_shape().as_list()
    batch_size = tf.shape(images)[0]
    images = tf.reshape(images, [-1, width*height, channel])
    samples = []

    for _ in range(n_samples):
        indices = tf.random_uniform(shape=[1], minval=0, maxval=width*height, dtype=tf.int32)
        selected_points = tf.gather(params=images, indices=indices, axis=1)
        samples.append(selected_points)

    samples = tf.concat(samples, axis=1)
    distances = 1e-4 * tf.ones(shape=(batch_size, n_samples))
    centroids = []
    for _ in range(n_clusters):
        indices = tf.reshape(tf.multinomial(sigma * distances, 1), [batch_size])
        weights = tf.expand_dims(tf.one_hot(indices, depth=n_samples), 2)
        selected_points = tf.expand_dims(tf.reduce_sum(weights * samples, axis=1), 1)
        centroids.append(selected_points)
    return tf.concat(centroids, axis=1)


def rgb_clustering(images, centroids, alpha):
    _, width, height, channel = images.get_shape().as_list()
    # Gaussian mixture clustering
    cluster_num = centroids.get_shape().as_list()[1]
    reshaped_images = tf.reshape(images, [-1, width, height, 1, channel])
    reshaped_centroids = tf.reshape(centroids, [-1, 1, 1, cluster_num, channel])
    distances = tf.reduce_sum(tf.square(reshaped_centroids - reshaped_images), axis=4)
    logits = tf.clip_by_value(-alpha * distances, -200, 200)
    probs = tf.expand_dims(tf.nn.softmax(logits), 4)
    new_images = tf.reduce_sum(reshaped_centroids * probs, axis=3)
    # update cluster centers
    new_centroids = tf.reduce_sum(reshaped_images * probs, axis=[1, 2]) / (tf.reduce_sum(probs, axis=[1, 2]) + 1e-16)
    return new_images, new_centroids


if __name__ == '__main__':
    from inception_v3 import download
    download(MODEL_PATH)