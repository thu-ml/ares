import os
import sys
import tensorflow as tf
import numpy as np

from realsafe.model.loader import load_model_from_path
from realsafe.dataset import imagenet, dataset_to_iterator

batch_size = 10

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

MODELS = [
    '../example/imagenet/inception_v3.py',
    '../example/imagenet/ens4_adv_inception_v3.py',
    '../example/imagenet/resnet_v2_alp.py',
    '../example/imagenet/resnet152_fd.py',
    '../example/imagenet/inception_v3_jpeg.py',
    '../example/imagenet/inception_v3_bit.py',
    '../example/imagenet/inception_v3_rand.py',
    '../example/imagenet/inception_v3_randmix.py',
]

rs = dict()
for model_path_short in MODELS:
    print('Loading {}...'.format(model_path_short))
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path_short)
    model = load_model_from_path(model_path).load(session)
    dataset = imagenet.load_dataset_for_classifier(model, offset=0, load_target=True).take(1000)
    xs_ph = tf.placeholder(model.x_dtype, shape=(None, *model.x_shape))
    labels = model.labels(xs_ph)

    accs = []
    for _ in range(10):
        for i_batch, (_, xs, ys, ys_target) in enumerate(dataset_to_iterator(dataset.batch(batch_size), session)):
            predictions = session.run(labels, feed_dict={xs_ph: xs})
            acc = np.equal(predictions, ys).astype(np.float32).mean()
            accs.append(acc)
            print('n={}..{} acc={:3f}'.format(i_batch * batch_size, i_batch * batch_size + batch_size - 1, acc))
    rs[model_path_short] = np.mean(accs)
    print('{} acc={:f}'.format(model_path, rs[model_path_short]))

for k, v in rs.items():
    print('{} acc={:f}'.format(k, v))
