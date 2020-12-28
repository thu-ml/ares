import tensorflow as tf
import numpy as np

from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph


class DeepFool(BatchAttack):
    ''' DeepFool. A white-box iterative optimization method. It needs to calculate the Jacobian of the logits with
    relate to input, so that it only applies to tasks with small number of classification class.

    - Supported distance metric: ``l_2``, ``l_inf``.
    - Supported goal: ``ut``.
    - References: https://arxiv.org/abs/1511.04599.
    '''

    def __init__(self, model, batch_size, distance_metric, session, iteration_callback=None):
        ''' Initialize DeepFool.

        :param model: The model to attack. A ``ares.model.ClassifierWithLogits`` instance.
        :param batch_size: Batch size for the ``batch_attack()`` method.
        :param distance_metric: Adversarial distance metric. All supported values are ``'l_2'`` and ``'l_inf'``.
        :param session: The ``tf.Session`` to run the attack in. The ``model`` should be loaded into this session.
        :param iteration_callback: A function accept a ``xs`` ``tf.Tensor`` (the original examples) and a ``xs_adv``
            ``tf.Tensor`` (the adversarial examples for ``xs``). During ``batch_attack()``, this callback function would
            be runned after each iteration, and its return value would be yielded back to the caller. By default,
            ``iteration_callback`` is ``None``.
        '''
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
        # we need to calculate the jacobian step by step
        self.grads_var = tf.Variable(tf.zeros((self.batch_size, self.model.n_class, np.prod(self.model.x_shape)),
                                              dtype=self.model.x_dtype))
        # calculating jacobian would construct a large graph
        self.assign_grads = [self.grads_var[:, i, :].assign(tf.gradients(logits[:, i], self.xs_adv_var)[0])
                             for i in range(self.model.n_class)]
        # get the target label's logits and jacobian
        k0s = tf.stack((tf.range(self.batch_size), self.ys_var), axis=1)
        yk0s = tf.expand_dims(tf.gather_nd(logits, k0s), axis=1)
        gradk0s = tf.expand_dims(tf.gather_nd(self.grads_var, k0s), axis=1)

        fs = tf.abs(yk0s - logits)
        ws = self.grads_var - gradk0s

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

        # if the xs_adv is adversarial, we do early stop.
        self.eqs = tf.equal(self.labels, self.ys_var)
        flags = tf.reshape(tf.cast(self.eqs, self.model.x_dtype) * (1 + self.overshot), (self.batch_size, 1))
        xs_adv_next = self.xs_adv_var + flags * rs
        xs_adv_next = tf.clip_by_value(xs_adv_next, self.model.x_min, self.model.x_max)

        self.update_xs_adv_step = self.xs_adv_var.assign(xs_adv_next)
        self.setup = [
            self.grads_var.initializer,
            self.xs_var.assign(tf.reshape(self.xs_ph, self.xs_var.shape)),
            self.xs_adv_var.assign(tf.reshape(self.xs_ph, self.xs_adv_var.shape)),
            self.ys_var.assign(self.ys_ph),
        ]
        self.setup_overshot = self.overshot.assign(self.overshot_ph)

        self.iteration_callback = None
        if iteration_callback is not None:
            xs_model = tf.reshape(self.xs_var, (self.batch_size, *self.model.x_shape))
            xs_adv_model = tf.reshape(self.xs_adv_var, (self.batch_size, *self.model.x_shape))
            self.iteration_callback = iteration_callback(xs_model, xs_adv_model)

        self.iteration = None
        self.details = {}

    def config(self, **kwargs):
        ''' (Re)config the attack.

        :param iteration: Iteration count. An integer.
        :param overshot: Overshot rate. A float number. Set to 0.02 by default.
        '''
        if 'iteration' in kwargs:
            self.iteration = kwargs['iteration']
        if 'overshot' in kwargs:
            self._session.run(self.setup_overshot, feed_dict={self.overshot_ph: kwargs['overshot']})

    def _batch_attack_generator(self, xs, ys, _):
        ''' Attack a batch of examples. It is a generator which yields back ``iteration_callback()``'s return value
        after each iteration if the ``iteration_callback`` is not ``None``, and returns the adversarial examples.
        '''
        self._session.run(self.setup, feed_dict={self.xs_ph: xs, self.ys_ph: ys})

        for _ in range(self.iteration):
            for assign_grad in self.assign_grads:
                self._session.run(assign_grad)
            self._session.run(self.update_xs_adv_step)
            succ = np.logical_not(self._session.run(self.eqs))
            if self.iteration_callback is not None:
                yield self._session.run(self.iteration_callback)
            if np.all(succ):  # early stop
                break

        self.details['success'] = succ

        return self._session.run(self.xs_adv_var).reshape((self.batch_size,) + self.model.x_shape)

    def batch_attack(self, xs, ys=None, ys_target=None):
        ''' Attack a batch of examples.

        :return: When the ``iteration_callback`` is ``None``, return the generated adversarial examples. When the
            ``iteration_callback`` is not ``None``, return a generator, which yields back the callback's return value
            after each iteration and returns the generated adversarial examples.
        '''
        g = self._batch_attack_generator(xs, ys, ys_target)
        if self.iteration_callback is None:
            try:
                next(g)
            except StopIteration as exp:
                return exp.value
        else:
            return g
