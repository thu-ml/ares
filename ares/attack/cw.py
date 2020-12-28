import tensorflow as tf
import numpy as np

from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph, maybe_to_array, scale
from ares.loss import CWLoss


class CW(BatchAttack):
    ''' Carlini & Wagner Attack (C&W). A white-box iterative optimization-based method. Require a differentiable logits.

    - Supported distance metric: ``l_2``.
    - Supported goal: ``t``, ``tm``, ``ut``.
    - References: https://arxiv.org/pdf/1608.04644.pdf.
    '''

    def __init__(self, model, batch_size, goal, distance_metric, session,
                 cw_loss_c=99999.0, confidence=0.0, learning_rate=0.01):
        ''' Initialize CW.

        :param model: The model to attack. A ``ares.model.ClassifierWithLogits`` instance.
        :param batch_size: Batch size for the ``batch_attack()`` method.
        :param goal: Adversarial goals. All supported values are ``'t'``, ``'tm'``, and ``'ut'``.
        :param session: The ``tf.Session`` to run the attack in. The ``model`` should be loaded into this session.
        :param cw_loss_c: The ``c`` parameter for ``ares.loss.CWLoss``.
        :param confidence: The minimum margin between the target logit and the second largest logit that we consider the
            example as adversarial.
        :param learning_rate: Learning rate for the ``AdamOptimizer``.
        '''
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
            raise NotImplementedError
        # the distance term
        if self.distance_metric == 'l_2':
            self.dists = tf.reduce_sum(tf.square(self.xs_adv - xs_var), axis=1)
        else:
            raise NotImplementedError
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
        self.logger = None

    def config(self, **kwargs):
        ''' (Re)config the attack.

        :param cs: Initial c, could be either a float number or a numpy float number array with shape of
            ``(self.batch_size,)``.
        :param iteration: An integer, represent iteration count for each search step and binary search step.
        :param search_steps: An integer, the number of times for running initial search for c before binary search.
        :param binsearch_steps: An integer, the number of times for running binary search on c.
        :param logger: A standard logger for logging verbose information during attacking.
        '''
        if 'cs' in kwargs:
            self.cs = maybe_to_array(kwargs['cs'], target_len=self.batch_size).astype(self.model.x_dtype.as_numpy_dtype)
        if 'iteration' in kwargs:
            self.iteration = kwargs['iteration']
        if 'search_steps' in kwargs:
            self.search_steps = kwargs['search_steps']
        if 'binsearch_steps' in kwargs:
            self.binsearch_steps = kwargs['binsearch_steps']

        if 'logger' in kwargs:
            self.logger = kwargs['logger']

    def batch_attack(self, xs, ys=None, ys_target=None):
        ''' Attack a batch of examples.

        :return: The generated adversarial examples.
        '''
        ys_input = ys_target if self.goal == 't' or self.goal == 'tm' else ys

        # create numpy index for fetching the original label's logit value
        ys_flatten = np.arange(0, self.batch_size * self.model.n_class, self.model.n_class)
        ys_flatten = ys_flatten.astype(self.model.y_dtype.as_numpy_dtype) + ys

        # store the adversarial examples and its distance to the original examples
        xs_adv = np.array(xs).astype(self.model.x_dtype.as_numpy_dtype).copy()
        min_dists = np.repeat(self.model.x_dtype.max, self.batch_size).astype(self.model.x_dtype.as_numpy_dtype)

        self._session.run(self.setup_xs, feed_dict={self.xs_ph: xs})
        self._session.run(self.setup_ys, feed_dict={self.ys_ph: ys_input})
        self._session.run(self.setup_d_ws)

        # setup initial cs value
        cs = self.cs.copy()
        self._session.run(self.setup_cs, feed_dict={self.cs_ph: cs})

        # find cs to begin with
        found = np.repeat(False, self.batch_size)
        for search_step in range(self.search_steps):
            # reset optimizer on each search step
            self._session.run(self.setup_optimizer)
            for _ in range(self.iteration):
                self._session.run(self.optimizer_step)
                new_score, new_logits, new_xs_adv, new_dists = self._session.run(
                    (self.score, self.logits, self.xs_adv_model, self.dists))
                better_dists = new_dists < min_dists
                if self.goal == 'ut' or self.goal == 'tm':
                    # for ut and tm goal, if the original label's logit is not the largest one, the example is
                    # adversarial.
                    new_succ = new_logits.max(axis=1) - new_logits.take(ys_flatten) > self.confidence
                else:
                    # for t goal, if the score is smaller than 0, the example is adversarial. The confidence is already
                    # included in the score, no need to add the confidence term here.
                    new_succ = new_score < 1e-12
                # if the example is adversarial and has small distance to the original example, update xs_adv and
                # min_dists
                to_update = np.logical_and(new_succ, better_dists)
                xs_adv[to_update], min_dists[to_update] = new_xs_adv[to_update], new_dists[to_update]
                found[to_update] = True

            if self.logger:
                self.logger.info('search_step={}, cs_mean={}, success_rate={}'.format(
                    search_step, cs.mean(), found.astype(np.float).mean()))

            if np.all(found):  # we have found an adversarial example for all inputs
                break
            else:  # update c value for all failed-to-attack inputs
                cs[np.logical_not(found)] *= 10.0
                self._session.run(self.setup_cs, feed_dict={self.cs_ph: cs})

        # prepare cs for binary search, no need to copy cs here
        cs_lo, cs_hi = np.zeros_like(cs), cs
        cs = (cs_hi + cs_lo) / 2
        self._session.run(self.setup_cs, feed_dict={self.cs_ph: cs})

        # binary search on cs
        for binsearch_step in range(self.binsearch_steps):
            # reset optimizer on each search step
            self._session.run(self.setup_optimizer)
            succ = np.repeat(False, self.batch_size)
            for _ in range(self.iteration):
                self._session.run(self.optimizer_step)
                new_score, new_logits, new_xs_adv, new_dists = self._session.run(
                    (self.score, self.logits, self.xs_adv_model, self.dists))
                better_dists = new_dists <= min_dists
                if self.goal == 'ut' or self.goal == 'tm':
                    # for ut and tm goal, if the original label's logit is not the largest one, the example is
                    # adversarial.
                    new_succ = new_logits.max(axis=1) - new_logits.take(ys_flatten) > self.confidence
                else:
                    # for t goal, if the score is smaller than 0, the example is adversarial. The confidence is already
                    # included in the score, no need to add the confidence term here.
                    new_succ = new_score < 1e-12
                # if the example is adversarial and has small distance to the original example, update xs_adv and
                # min_dists
                to_update = np.logical_and(new_succ, better_dists)
                xs_adv[to_update], min_dists[to_update] = new_xs_adv[to_update], new_dists[to_update]
                succ[to_update] = True
                # the initial search for c might fail, while we might succeed finding an adversarial example during
                # binary search
                found[to_update] = True

            # update cs
            not_succ = np.logical_not(succ)
            cs_lo[not_succ], cs_hi[succ] = cs[not_succ], cs[succ]
            cs = (cs_hi + cs_lo) / 2.0
            self._session.run(self.setup_cs, feed_dict={self.cs_ph: cs})

            if self.logger:
                self.logger.info('binsearch_step={}, cs_mean={}, success_rate={}'.format(
                    binsearch_step, cs.mean(), succ.astype(np.float).mean()))

        self.details['success'] = found
        return xs_adv

    def _scale_to_model(self, xs):
        return scale(xs, self.model.x_min, self.model.x_max, -1.0, 1.0)

    def _scale_to_tanh(self, xs):
        # np.arctanh(np.tanh(np.arctanh(1.0 - 1e-6) + 10.0)) == 17.242754385535303
        bound = 1.0 - 1e-6
        return scale(xs, -bound, bound, self.model.x_min, self.model.x_max)
