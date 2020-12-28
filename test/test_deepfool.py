import tensorflow as tf
import numpy as np
import os

from keras.datasets.cifar10 import load_data

from ares import DeepFool
from ares.model.loader import load_model_from_path

batch_size = 100

session = tf.Session()

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../example/cifar10/resnet56.py')
rs_model = load_model_from_path(model_path)
model = rs_model.load(session)

_, (xs_test, ys_test) = load_data()
xs_test = (xs_test / 255.0) * (model.x_max - model.x_min) + model.x_min
ys_test = ys_test.reshape(len(ys_test))

xs_ph = tf.placeholder(model.x_dtype, shape=(batch_size, *model.x_shape))
lgs, lbs = model.logits_and_labels(xs_ph)

xs = xs_test[0:batch_size]
ys = ys_test[0:batch_size]


def iteration_callback(xs, xs_adv):
    delta = tf.abs(xs_adv - xs)
    return tf.reduce_max(tf.reshape(delta, (xs.shape[0], -1)), axis=1)


attack = DeepFool(
    model=model,
    batch_size=batch_size,
    distance_metric='l_2',
    session=session,
    iteration_callback=iteration_callback,
)
attack.config(
    overshot=0.02,
    iteration=50,
)

try:
    g = attack.batch_attack(xs, ys=ys)
    while True:
        dists = next(g)
        print(dists.min(), dists.max(), dists.mean())
except StopIteration as e:
    xs_adv = e.value

lbs_pred = session.run(lbs, feed_dict={xs_ph: xs})
lbs_adv = session.run(lbs, feed_dict={xs_ph: xs_adv})

print(
    np.equal(ys, lbs_pred).astype(np.float).mean(),
    np.equal(ys, lbs_adv).astype(np.float).mean()
)
