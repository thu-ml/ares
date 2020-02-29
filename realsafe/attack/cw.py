import tensorflow as tf
import numpy as np

from realsafe.attack.base import BatchAttack
from realsafe.attack.utils import get_xs_ph, get_ys_ph, maybe_to_array, scale
from realsafe.loss import CWLoss


class CW(BatchAttack):
    '''
    Carlini & Wagner Attack (C&W)
    A white-box iterative optmization-based method. Require a differentiable logits.

    Supported distance metric: `l_2`
    Supported goal: `t`, `tm`, `ut`
    Supported config parameters:
    - TODO

    References:
    [1] TODO
    '''

    def __init__(self, model, batch_size, goal, distance_metric, session,
                 cw_loss_c=99999.0, confidence=0.0, learning_rate=0.01):
        self.model, self.batch_size, self._session = model, batch_size, session
        self.goal, self.distance_metric = goal, distance_metric
        self.confidence = confidence

        # flatten shape of xs_ph
        xs_shape_flatten = (self.batch_size, np.prod(self.model.x_shape))
        # placeholder for batch_attack's input
        self.xs_ph, self.ys_ph = get_xs_ph(model, self.batch_size), get_ys_ph(model, self.batch_size)
        # store adversarial examples and labels in variables to reduce memory copy between tensorflow and python
        xs_var = tf.Variable(tf.zeros(shape=xs_shape_flatten, dtype=self.model.x_dtype))
        ys_var = tf.Variable(tf.zeros_like(self.ys_ph))
        # placeholder for c
        self.cs_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        cs_var = tf.Variable(tf.zeros_like(self.cs_ph))
        # xs_adv = tanh(ws)
        d_ws = tf.Variable(tf.zeros(shape=xs_shape_flatten, dtype=self.model.x_dtype))
        ws = tf.atanh(self._scale_to_tanh(xs_var)) + d_ws
        self.xs_adv = self._scale_to_model(tf.tanh(ws))
        self.xs_adv_model = tf.reshape(self.xs_adv, (self.batch_size, *self.model.x_shape))
        # the C&W loss term
        cw_loss = CWLoss(self.model)(self.xs_adv_model, ys_var)
        self.logits = self.model.logits(self.xs_adv_model)
        if self.goal == 't' or self.goal == 'tm':
            self.score = tf.maximum(0.0, cw_loss + confidence)
        elif self.goal == 'ut':
            self.score = tf.maximum(0.0, tf.negative(cw_loss) + confidence)
        else:
            raise NotImplemented
        # the distance term
        if self.distance_metric == 'l_2':
            self.dists = tf.reduce_sum(tf.square(self.xs_adv - xs_var), axis=1)
        else:
            raise NotImplemented
        # the loss
        loss = self.dists + cs_var * self.score
        # minimize the loss using Adam
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.optimizer_step = optimizer.minimize(loss, var_list=[d_ws])
        self.setup_optimizer = tf.variables_initializer(optimizer.variables())

        self.setup_xs = xs_var.assign(tf.reshape(self.xs_ph, xs_shape_flatten))
        self.setup_ys = ys_var.assign(self.ys_ph)
        self.setup_cs = cs_var.assign(self.cs_ph)
        self.setup_d_ws = d_ws.assign(tf.zeros_like(d_ws))

        # provides default values
        self.iteration = 50
        self.search_steps = 2
        self.binsearch_steps = 10

        self.details = {}

    def config(self, **kwargs):
        if 'cs' in kwargs:
            self.cs = maybe_to_array(kwargs['cs'], target_len=self.batch_size).astype(self.model.x_dtype.as_numpy_dtype)
        if 'iteration' in kwargs:
            self.iteration = kwargs['iteration']
        if 'search_steps' in kwargs:
            self.search_steps = kwargs['search_steps']
        if 'binsearch_steps' in kwargs:
            self.binsearch_steps = kwargs['binsearch_steps']

    def batch_attack(self, xs, ys=None, ys_target=None):
        ys_flatten = np.arange(0, self.batch_size * self.model.n_class, self.model.n_class) + ys
        ys_input = ys_target if self.goal == 't' or self.goal == 'tm' else ys
        cs, xs_adv = self.cs.copy(), np.array(xs).copy()
        self._session.run((self.setup_xs, self.setup_ys, self.setup_d_ws), feed_dict={
            self.xs_ph: xs, self.ys_ph: ys_input
        })
        # find c to begin with
        found = np.repeat(False, self.batch_size)
        min_dists = np.repeat(self.model.x_dtype.max, self.batch_size)
        for _ in range(self.search_steps):
            # reset optimizer on each search step
            self._session.run(self.setup_optimizer)
            self._session.run(self.setup_cs, feed_dict={self.cs_ph: cs})
            for _ in range(self.iteration):
                self._session.run(self.optimizer_step)
                new_score, new_logits, new_xs_adv, new_dists = self._session.run(
                    (self.score, self.logits, self.xs_adv_model, self.dists)
                )
                if self.goal == 'ut' or self.goal == 'tm':
                    new_succ = new_logits.max(axis=1) - new_logits.take(ys_flatten) > self.confidence
                else:
                    new_succ = new_score < 1e-12
                better_dists = new_dists <= min_dists
                to_update = np.logical_and(new_succ, better_dists)
                xs_adv[to_update], min_dists[to_update] = new_xs_adv[to_update], new_dists[to_update]
                found[to_update] = True
            if np.all(found):
                break
            else:
                cs[np.logical_not(found)] *= 10.0

        cs_hi = cs
        cs_lo = np.zeros_like(cs)
        cs = (cs_hi + cs_lo) / 2

        # binsearch
        for i in range(self.binsearch_steps):
            # reset optimizer on each search step
            self._session.run(self.setup_optimizer)
            self._session.run(self.setup_cs, feed_dict={self.cs_ph: cs})

            succ = np.repeat(False, self.batch_size)

            for _ in range(self.iteration):
                self._session.run(self.optimizer_step)
                new_score, new_logits, new_xs_adv, new_dists = self._session.run(
                    (self.score, self.logits, self.xs_adv_model, self.dists)
                )
                if self.goal == 'ut' or self.goal == 'tm':
                    new_succ = new_logits.max(axis=1) - new_logits.take(ys_flatten) > self.confidence
                else:
                    new_succ = new_score < 1e-12
                better_dists = new_dists <= min_dists
                to_update = np.logical_and(new_succ, better_dists)
                xs_adv[to_update], min_dists[to_update] = new_xs_adv[to_update], new_dists[to_update]
                succ[to_update] = True

            not_succ = np.logical_not(succ)
            cs_hi[succ] = cs[succ]
            cs_lo[not_succ] = cs[not_succ]
            cs = (cs_hi + cs_lo) / 2.0

        self.details['success'] = found
        return xs_adv

    def _scale_to_model(self, xs):
        return scale(xs, self.model.x_min, self.model.x_max, -1.0, 1.0)

    def _scale_to_tanh(self, xs):
        # np.arctanh(np.tanh(np.arctanh(1.0 - 1e-6) + 10.0)) == 17.242754385535303
        bound = 1.0 - 1e-6
        return scale(xs, -bound, bound, self.model.x_min, self.model.x_max)
