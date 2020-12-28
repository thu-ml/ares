import tensorflow as tf
import numpy as np
import os

from ares.dataset import cifar10, dataset_to_iterator
from ares.model.loader import load_model_from_path
from ares.model.ensemble import EnsembleModel, EnsembleRandomnessModel

batch_size = 100

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../example/cifar10/resnet56.py')
model = load_model_from_path(model_path).load(session)
e_model = EnsembleModel([model, model], [0.5, 0.5])
er_model = EnsembleRandomnessModel(model, 10, session)

ds = cifar10.load_dataset_for_classifier(model).batch(batch_size).take(1)
_, xs, ys = next(dataset_to_iterator(ds, session))

xs_ph = tf.placeholder(model.x_dtype, shape=(batch_size, *model.x_shape))

labels = model.labels(xs_ph)
e_labels = e_model.labels(xs_ph)
er_labels = er_model.labels(xs_ph)

labels_np = session.run(labels, feed_dict={xs_ph: xs})
e_labels_np = session.run(e_labels, feed_dict={xs_ph: xs})
er_labels_np = session.run(er_labels, feed_dict={xs_ph: xs})

assert(np.array_equal(labels_np, e_labels_np))
assert(np.array_equal(labels_np, er_labels_np))

print(labels_np)
print(e_labels_np)
print(er_labels_np)