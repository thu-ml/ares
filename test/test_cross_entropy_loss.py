import tensorflow as tf
import numpy as np
import os

from ares import BIM, CrossEntropyLoss, EnsembleCrossEntropyLoss, EnsembleRandomnessCrossEntropyLoss
from ares.dataset import cifar10, dataset_to_iterator
from ares.model.loader import load_model_from_path

batch_size = 1000

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../example/cifar10/resnet56.py')
model = load_model_from_path(model_path).load(session)

loss_op = CrossEntropyLoss(model)
e_loss_op = EnsembleCrossEntropyLoss([model, model], [0.5, 0.5])
er_loss_op = EnsembleRandomnessCrossEntropyLoss(model, 10, session)

ds = cifar10.load_dataset_for_classifier(model).batch(batch_size).take(1)
_, xs, ys = next(dataset_to_iterator(ds, session))

xs_ph = tf.placeholder(model.x_dtype, shape=(batch_size, *model.x_shape))
ys_ph = tf.placeholder(model.y_dtype, shape=batch_size)

loss = loss_op(xs_ph, ys_ph)
e_loss = e_loss_op(xs_ph, ys_ph)
er_loss = er_loss_op(xs_ph, ys_ph)

dloss_dxs = tf.gradients(loss, xs_ph)[0]
de_loss_dxs = tf.gradients(e_loss, xs_ph)[0]
der_loss_dxs = tf.gradients(er_loss, xs_ph)[0]

loss_np, dloss_dxs_np = session.run((loss, dloss_dxs), feed_dict={xs_ph: xs, ys_ph: ys})
e_loss_np, de_loss_dxs_np = session.run((e_loss, de_loss_dxs), feed_dict={xs_ph: xs, ys_ph: ys})
er_loss_np, der_loss_dxs_np = session.run((er_loss, der_loss_dxs), feed_dict={xs_ph: xs, ys_ph: ys})

e0 = np.abs(loss_np - e_loss_np).max()
e1 = np.abs(loss_np - er_loss_np).max()
e2 = np.abs(dloss_dxs_np - de_loss_dxs_np).max()
e3 = np.abs(dloss_dxs_np - der_loss_dxs_np).max()

print(e0, e1, e2, e3)

assert(e0 < 1e-3)
assert(e1 < 1e-3)
assert(e2 < 1e-3)
assert(e3 < 1e-3)
