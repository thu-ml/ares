from ares.model.loader import load_model_from_path
from ares import NAttack, CWLoss

from keras.datasets.cifar10 import load_data

import os
import numpy as np
import tensorflow as tf

batch_size = 20

session = tf.Session()

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../example/cifar10/resnet56.py')
rs_model = load_model_from_path(model_path)
model = rs_model.load(session)

_, (xs_test, ys_test) = load_data()
xs_test = (xs_test / 255.0) * (model.x_max - model.x_min) + model.x_min
ys_test = ys_test.reshape(len(ys_test))

xs_ph = tf.placeholder(model.x_dtype, shape=(1, *model.x_shape))
lgs, lbs = model.logits_and_labels(xs_ph)

MAGNITUDE = 0.5 * (model.x_max - model.x_min)

loss = CWLoss(model)
attack = NAttack(
    model=model,
    loss=loss,
    goal='ut',
    distance_metric='l_2',
    session=session,
    samples_per_draw=100,
    samples_batch_size=25,
    dimension_reduction=(28, 28),
)
logger = tf.get_logger()
logger.setLevel(tf.logging.INFO)
attack.config(
    max_queries=20000,
    magnitude=MAGNITUDE,
    sigma=0.1,
    lr=0.02,
    logger=logger,
)

xs_ph = tf.placeholder(model.x_dtype, shape=(batch_size, *model.x_shape))
lgs, lbs = model.logits_and_labels(xs_ph)

xs, ys = xs_test[:batch_size], ys_test[:batch_size]
xs_adv = []
for i in range(batch_size):
    print(i, end=' ')
    x, y = xs[i], ys[i]
    x_adv = attack.attack(x, y=y)
    print(np.linalg.norm(x_adv - x), end=' ')
    xs_adv.append(x_adv)
    print(attack.details)

xs_adv = np.array(xs_adv)

lbs_pred = session.run(lbs, feed_dict={xs_ph: xs})
lbs_adv = session.run(lbs, feed_dict={xs_ph: xs_adv})

print(
    np.equal(ys, lbs_pred).astype(np.float).mean(),
    np.equal(ys, lbs_adv).astype(np.float).mean()
)
