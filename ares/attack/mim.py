import tensorflow as tf
import numpy as np

from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph, maybe_to_array, get_unit


class MIM(BatchAttack):
    ''' Momentum Iterative Method (MIM). A white-box iterative constraint-based method. Require a differentiable loss
    function and a ``ares.model.Classifier`` model.

    - Supported distance metric: ``l_2``, ``l_inf``.
    - Supported goal: ``t``, ``tm``, ``ut``.
    - References: https://arxiv.org/abs/1710.06081.
    '''

    def __init__(self, model, batch_size, loss, goal, distance_metric, session, iteration_callback=None):
        ''' Initialize MIM.

        :param model: The model to attack. A ``ares.model.Classifier`` instance.
        :param batch_size: Batch size for the ``batch_attack()`` method.
        :param loss: The loss function to optimize. A ``ares.loss.Loss`` instance.
        :param goal: Adversarial goals. All supported values are ``'t'``, ``'tm'``, and ``'ut'``.
        :param distance_metric: Adversarial distance metric. All supported values are ``'l_2'`` and ``'l_inf'``.
        :param session: The ``tf.Session`` to run the attack in. The ``model`` should be loaded into this session.
        :param iteration_callback: A function accept a ``xs`` ``tf.Tensor`` (the original examples) and a ``xs_adv``
            ``tf.Tensor`` (the adversarial examples for ``xs``). During ``batch_attack()``, this callback function would
            be runned after each iteration, and its return value would be yielded back to the caller. By default,
            ``iteration_callback`` is ``None``.
        '''
        self.model, self.batch_size, self._session = model, batch_size, session
        self.loss, self.goal, self.distance_metric = loss, goal, distance_metric
        # placeholder for batch_attack's input
        self.xs_ph = get_xs_ph(model, batch_size)
        self.ys_ph = get_ys_ph(model, batch_size)
        # flatten shape of xs_ph
        xs_flatten_shape = (batch_size, np.prod(self.model.x_shape))
        # store xs and ys in variables to reduce memory copy between tensorflow and python
        # variable for the original example with shape of (batch_size, D)
        self.xs_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))
        # variable for labels
        self.ys_var = tf.Variable(tf.zeros(shape=(batch_size,), dtype=self.model.y_dtype))
        # variable for the (hopefully) adversarial example with shape of (batch_size, D)
        self.xs_adv_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))
        # variable for the momentum term
        self.g_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))
        # decay factor
        self.decay_factor_ph = tf.placeholder(self.model.x_dtype, ())
        self.decay_factor_var = tf.Variable(tf.zeros(shape=(), dtype=self.model.x_dtype))
        # magnitude
        self.eps_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.eps_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))
        # step size
        self.alpha_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.alpha_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))
        # expand dim for easier broadcast operations
        eps = tf.expand_dims(self.eps_var, 1)
        alpha = tf.expand_dims(self.alpha_var, 1)
        # calculate loss' gradient with relate to the adversarial example
        # grad.shape == (batch_size, D)
        self.xs_adv_model = tf.reshape(self.xs_adv_var, (batch_size, *self.model.x_shape))
        self.loss = loss(self.xs_adv_model, self.ys_var)
        grad = tf.gradients(self.loss, self.xs_adv_var)[0]
        if goal == 't' or goal == 'tm':
            grad = -grad
        elif goal != 'ut':
            raise NotImplementedError
        # 1-norm of gradient
        grad_l1 = tf.reduce_sum(tf.abs(grad), axis=1)
        # update the momentum term
        g_next = self.decay_factor_var * self.g_var + grad / tf.expand_dims(grad_l1, 1)
        self.update_g_step = self.g_var.assign(g_next)
        # update the adversarial example
        if distance_metric == 'l_2':
            g_unit = get_unit(self.g_var)
            xs_adv_delta = self.xs_adv_var - self.xs_var + alpha * g_unit
            # clip by max l_2 magnitude of adversarial noise
            xs_adv_next = self.xs_var + tf.clip_by_norm(xs_adv_delta, eps, axes=[1])
        elif distance_metric == 'l_inf':
            xs_lo, xs_hi = self.xs_var - eps, self.xs_var + eps
            g_sign = tf.sign(self.g_var)
            # clip by max l_inf magnitude of adversarial noise
            xs_adv_next = tf.clip_by_value(self.xs_adv_var + alpha * g_sign, xs_lo, xs_hi)
        else:
            raise NotImplementedError
        # clip by (x_min, x_max)
        xs_adv_next = tf.clip_by_value(xs_adv_next, self.model.x_min, self.model.x_max)
        self.update_xs_adv_step = self.xs_adv_var.assign(xs_adv_next)

        self.config_eps_step = self.eps_var.assign(self.eps_ph)
        self.config_alpha_step = self.alpha_var.assign(self.alpha_ph)
        self.config_decay_factor_step = self.decay_factor_var.assign(self.decay_factor_ph)
        self.setup_xs = [self.xs_var.assign(tf.reshape(self.xs_ph, xs_flatten_shape)),
                         self.xs_adv_var.assign(tf.reshape(self.xs_ph, xs_flatten_shape))]
        self.setup_ys = self.ys_var.assign(self.ys_ph)
        self.setup_g = tf.variables_initializer([self.g_var])
        self.iteration = None

        self.iteration_callback = None
        if iteration_callback is not None:
            xs_model = tf.reshape(self.xs_var, (self.batch_size, *self.model.x_shape))
            self.iteration_callback = iteration_callback(xs_model, self.xs_adv_model)

    def config(self, **kwargs):
        ''' (Re)config the attack.

        :param magnitude: Max distortion, could be either a float number or a numpy float number array with shape of
            ``(self.batch_size,)``.
        :param alpha: Step size for each iteration, could be either a float number or a numpy float number array with
            shape of ``(self.batch_size,)``.
        :param decay_factor: A float number, the decay factor for momentum.
        :param iteration: An integer, the iteration count.
        '''
        if 'magnitude' in kwargs:
            eps = maybe_to_array(kwargs['magnitude'], self.batch_size)
            self._session.run(self.config_eps_step, feed_dict={self.eps_ph: eps})
        if 'alpha' in kwargs:
            alpha = maybe_to_array(kwargs['alpha'], self.batch_size)
            self._session.run(self.config_alpha_step, feed_dict={self.alpha_ph: alpha})
        if 'decay_factor' in kwargs:
            decay_factor = kwargs['decay_factor']
            self._session.run(self.config_decay_factor_step, feed_dict={self.decay_factor_ph: decay_factor})
        if 'iteration' in kwargs:
            self.iteration = kwargs['iteration']

    def _batch_attack_generator(self, xs, ys, ys_target):
        labels = ys if self.goal == 'ut' else ys_target
        self._session.run(self.setup_xs, feed_dict={self.xs_ph: xs})
        self._session.run(self.setup_ys, feed_dict={self.ys_ph: labels})
        self._session.run(self.setup_g)
        for _ in range(self.iteration):
            self._session.run(self.update_g_step)
            self._session.run(self.update_xs_adv_step)
            if self.iteration_callback is not None:
                yield self._session.run(self.iteration_callback)
        return self._session.run(self.xs_adv_model)

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
