import sys
import os
THIRD_PARTY_PATH = '../../third_party/models/research/slim'
THIRD_PARTY_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), THIRD_PARTY_PATH))
sys.path.append(THIRD_PARTY_PATH)

import tensorflow as tf

from nets import inception_v3

from realsafe import ClassifierWithLogits
from realsafe.model.loader import get_res_path

slim = tf.contrib.slim


def load(session):
    model = Ens4AdvInceptionV3()
    model.load(session)
    return model


class Ens4AdvInceptionV3(ClassifierWithLogits):
    def __init__(self):
        ClassifierWithLogits.__init__(self, 1001, 0.0, 1.0, (299, 299, 3), tf.float32, tf.int32)

    def _logits_and_labels(self, xs_ph):
        xs_ph = xs_ph * 2.0 - 1.0

        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits, end_points = inception_v3.inception_v3(xs_ph, num_classes=self.n_class,
                                                           is_training=False, reuse=tf.AUTO_REUSE)

            predicted_labels = tf.cast(tf.argmax(end_points['Predictions'], 1), tf.int32)

        return logits, predicted_labels

    def load(self, session):
        x_input = tf.placeholder(self.x_dtype, shape=(None,) + self.x_shape)
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            inception_v3.inception_v3(x_input, num_classes=self.n_class, is_training=False, reuse=tf.AUTO_REUSE)

        model_path = get_res_path('./imagenet/ens4_adv_inception_v3')
        if not os.path.exists(model_path):
            import urllib
            import tarfile

            os.makedirs(model_path)
            urllib.request.urlretrieve(
                'http://download.tensorflow.org/models/ens4_adv_inception_v3_2017_08_18.tar.gz',
                os.path.join(model_path, 'ens4_adv_inception_v3_2017_08_18.tar.gz'))

            tar = tarfile.open(os.path.join(model_path, 'ens4_adv_inception_v3_2017_08_18.tar.gz'))
            file_names = tar.getnames()
            for file_name in file_names:
                tar.extract(file_name, model_path)

        saver = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        saver.restore(session, os.path.join(model_path, 'ens4_adv_inception_v3.ckpt'))
