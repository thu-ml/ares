import numpy as np
import tensorflow as tf
import os

from keras.datasets.cifar10 import load_data

from realsafe import Boundary
from realsafe.model.loader import load_model_from_path

logger = tf.get_logger()
logger.setLevel(tf.logging.INFO)

batch_size = 200

session = tf.Session()

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../example/cifar10/resnet56.py')
rs_model = load_model_from_path(model_path)
model = rs_model.load(session)

_, (xs_test, ys_test) = load_data()
xs_test = (xs_test / 255.0) * (model.x_max - model.x_min) + model.x_min
ys_test = ys_test.reshape(len(ys_test))

xs_ph = tf.placeholder(model.x_dtype, shape=(1, *model.x_shape))
lgs, lbs = model.logits_and_labels(xs_ph)


def iteration_callback(xs, xs_adv):
    delta = xs_adv - xs
    return tf.linalg.norm(tf.reshape(delta, (delta.shape[0], -1)), axis=1)


attack = Boundary(
    model=model,
    batch_size=batch_size,
    goal='ut',
    session=session,
    iteration_callback=iteration_callback,
)
attack.config(
    max_directions=25,
    max_queries=20000,
    spherical_step=1e-2,
    source_step=1e-2,
    step_adaptation=1.5,
    maxprocs=50,
    logger=logger,
)

xs_ph = tf.placeholder(model.x_dtype, shape=(None, *model.x_shape))
lgs, lbs = model.logits_and_labels(xs_ph)

xs, ys = xs_test[:batch_size], ys_test[:batch_size]
starting_points = []
for y in ys:
    while True:
        starting_point = np.random.uniform(model.x_min, model.x_max, size=model.x_shape)
        starting_point = starting_point.astype(model.x_dtype.as_numpy_dtype)
        if session.run(lbs, feed_dict={xs_ph: starting_point[np.newaxis]})[0] != y:
            starting_points.append(starting_point)
            break
starting_points = np.stack(starting_points)
attack.config(starting_points=starting_points)
try:
    g = attack.batch_attack(xs, ys=ys)
    while True:
        dists = next(g)
        print(dists.min(), dists.max(), dists.mean())
except StopIteration as exp:
    xs_adv = exp.value

lbs_pred = session.run(lbs, feed_dict={xs_ph: xs})
lbs_adv = session.run(lbs, feed_dict={xs_ph: xs_adv})

print([np.linalg.norm(x) for x in xs_adv - xs])
print([np.max(np.abs(x)) for x in xs_adv - xs])
print(np.equal(ys, lbs_pred).astype(np.float).mean(), np.equal(ys, lbs_adv).astype(np.float).mean())
