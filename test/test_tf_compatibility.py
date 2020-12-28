import tensorflow as tf
import numpy as np

from ares.attack.utils import clip_eta_batch

session = tf.Session()
batch_size = 100
x_shape = (28, 28, 1)

xs_ph = tf.placeholder(tf.float64, shape=(batch_size, *x_shape))
xs = tf.Variable(tf.zeros_like(xs_ph))
session.run(xs.assign(xs_ph), feed_dict={xs_ph: np.random.rand(batch_size, *x_shape)})

xs_flatten = tf.reshape(xs, (batch_size, -1))
xs_norm = tf.norm(xs_flatten, axis=1)
assert xs_norm.shape == (batch_size,)
xs_unit = xs_flatten / tf.expand_dims(xs_norm, 1)
assert xs_unit.shape == (batch_size, np.prod(x_shape))

xs_norm_bound = np.abs(np.random.rand(batch_size)) + 1.0
xs_clip = tf.clip_by_norm(xs_flatten, xs_norm_bound.reshape((batch_size, 1)), axes=[1])
xs_clip_norm = tf.norm(xs_clip, axis=1)
assert np.alltrue(np.less_equal(np.abs(session.run(xs_clip_norm) - xs_norm_bound), 1e-12))

xs_clip = clip_eta_batch(xs, 1.0, 'l_2')
xs_clip_norm = np.array([np.linalg.norm(x) for x in session.run(xs_clip)])
assert np.alltrue(np.less_equal(np.abs(xs_clip_norm - 1.0), 1e-12))