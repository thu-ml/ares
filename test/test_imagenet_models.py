import os
import sys
import tensorflow as tf
import numpy as np

from realsafe.model.loader import load_model_from_path

batch_size = 100

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

MODELS = [
    '../example/imagenet/inception_v3.py',
    '../example/imagenet/inception_v3_bit.py',
    '../example/imagenet/inception_v3_jpeg.py',
    '../example/imagenet/inception_v3_rand.py',
    '../example/imagenet/ens4_adv_inception_v3.py',
    '../example/imagenet/resnet152_fd.py',
    '../example/imagenet/resnet_v2_alp.py',
]

for model_path in MODELS:
    print('Loading {}...'.format(model_path))
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    model = load_model_from_path(model_path).load(session)
    xs_ph = tf.placeholder(model.x_dtype, shape=(10, *model.x_shape))
    session.run(model.logits(xs_ph), feed_dict={xs_ph: np.zeros((10, *model.x_shape))})
