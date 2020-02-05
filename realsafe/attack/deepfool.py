import tensorflow as tf
import numpy as np

from realsafe.attack.base import BatchAttack
from realsafe.attack.utils import get_xs_ph, get_ys_ph, maybe_to_array, get_unit


class DeepFool(BatchAttack):
    """
    DeepFool
    A white-box iterative optimization method. It needs to calculate the Jacobian of the logits with relate to input,
    so that it only applies to tasks with small number of classification class (This method does not use loss function).

    Supported distance metric: `l_2`, `l_inf`
    Supported goal: `ut`
    Supported config parameters:
    - `magnitude`: max distortion, could be either a float number or a numpy float number array with shape of
        (batch_size,).
    - `alpha`: step size for each iteration, could be either a float number or a numpy float number array with shape of
        (batch_size,).
    - `iteration`: an integer, represent iteration count.

    References:
    [1] https://arxiv.org/abs/1511.04599
    """

    def __init__(self, model, batch_size, goal, distance_metric, session):
        self.model, self.batch_size, self._session = model, batch_size, session
        self.overshot = tf.Variable(0.02)
        self.overshot_ph = tf.placeholder(tf.float32)
        # placeholder for batch_attack's input
        self.xs_ph = get_xs_ph(model, batch_size)
        self.ys_ph = get_ys_ph(model, batch_size)
        # store xs, xs_adv and ys in variables to reduce memory copy between tensorflow and python
        # flatten shape of xs_ph
        xs_flatten_shape = (batch_size, np.prod(self.model.x_shape))
        # variable for the original example with shape of (batch_size, D)
        self.xs_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))
        # variable for labels
        self.ys_var = tf.Variable(tf.zeros(shape=(batch_size,), dtype=self.model.y_dtype))
        # variable for the (hopefully) adversarial example with shape of (batch_size, D)
        self.xs_adv_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))
        # get the adversarial example's logits and labels
        logits, self.labels = self.model.logits_and_labels(
            xs=tf.reshape(self.xs_adv_var, (batch_size,) + self.model.x_shape))
        # calculating a jocobian would construct a large graph
        grads = [tf.gradients(logits[:, i], self.xs_adv_var)[0] for i in range(self.model.n_class)]
        grads = tf.stack(grads, axis=0)
        grads = tf.transpose(grads, (1, 0, 2))
        # get the target label's logits and jacobian
        k0s = tf.stack((tf.range(self.batch_size), self.ys_var), axis=1)
        yk0s = tf.expand_dims(tf.gather_nd(logits, k0s), axis=1)
        gradk0s = tf.expand_dims(tf.gather_nd(grads, k0s), axis=1)

        fs = tf.abs(yk0s - logits)
        ws = grads - gradk0s

        ws_norm = tf.norm(ws, axis=-1)
        # for index = k0, ws_norm = 0.0, fs = 0.0, ls = 0.0 / 0.0 = NaN, and tf.argmin would ignore NaN
        ls = fs / ws_norm
        ks = tf.argmin(ls, axis=1, output_type=self.model.y_dtype)
        ks = tf.stack((tf.range(self.batch_size), ks), axis=1)

        fsks = tf.gather_nd(fs, ks)
        ws_normks = tf.gather_nd(ws_norm, ks)
        if distance_metric == 'l_2':
            wsks = tf.gather_nd(ws, ks)
            rs = tf.reshape(fsks / tf.square(ws_normks), (self.batch_size, 1)) * wsks
        elif distance_metric == 'l_inf':
            ws_sign_ks = tf.gather_nd(tf.sign(ws), ks)
            rs = tf.reshape(fsks / ws_normks, (self.batch_size, 1)) * ws_sign_ks
        else:
            raise NotImplementedError

        eqs = tf.equal(self.labels, self.ys_var)
        self.flag = tf.reduce_any(eqs)
        flags = tf.reshape(tf.cast(eqs, self.model.x_dtype) * (1 + self.overshot), (self.batch_size, 1))

        xs_adv_next = self.xs_adv_var + flags * rs
        xs_adv_next = tf.clip_by_value(xs_adv_next, self.model.x_min, self.model.x_max)

        self.update_xs_adv_step = self.xs_adv_var.assign(xs_adv_next)
        self.setup = [
            self.xs_var.assign(tf.reshape(self.xs_ph, self.xs_var.shape)),
            self.xs_adv_var.assign(tf.reshape(self.xs_ph, self.xs_adv_var.shape)),
            self.ys_var.assign(self.ys_ph),
        ]
        self.setup_overshot = self.overshot.assign(self.overshot_ph)

        self.iteration = None

    def config(self, **kwargs):
        if 'iteration' in kwargs:
            self.iteration = kwargs['iteration']
        if 'overshot' in kwargs:
            self._session.run(self.setup_overshot, feed_dict={self.overshot_ph: kwargs['overshot']})

    def batch_attack(self, xs, ys=None, ys_target=None):
        self._session.run(self.setup, feed_dict={ self.xs_ph: xs, self.ys_ph: ys })

        for _ in range(self.iteration):
            self._session.run(self.update_xs_adv_step)
            flag = self._session.run(self.flag)
            if not flag:
                break

        return self._session.run(self.xs_adv_var).reshape((self.batch_size,) + self.model.x_shape)