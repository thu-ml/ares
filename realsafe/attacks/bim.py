import tensorflow as tf

from realsafe.attacks.base import BatchAttack
from realsafe.attacks.utils import get_xs_ph, get_ys_ph, maybe_to_array, get_unit


class BIM(BatchAttack):
    """
    TODO
    """

    def __init__(self, model, batch_size, loss, goal, distance_metric, session):
        self.model, self.batch_size, self._session = model, batch_size, session
        self.loss, self.goal, self.distance_metric = loss, goal, distance_metric
        # placeholder for batch_attack's input
        self.xs_ph = get_xs_ph(model, batch_size)
        self.ys_ph = get_ys_ph(model, batch_size)
        # variable for storing the original example
        self.xs_var = tf.Variable(tf.zeros_like(self.xs_ph))
        # variable for the adversarial noise
        self.delta_xs_var = tf.Variable(tf.zeros_like(self.xs_ph))
        # magnitude
        self.eps_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.eps_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))
        # step size
        self.alpha_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.alpha_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))
        # reshape to (batch_size, 1) for broadcast operations with gradient
        eps = tf.expand_dims(self.eps_var, 1)
        alpha = tf.expand_dims(self.alpha_var, 1)
        # calculate gradient according to adversary's goal
        if goal == 't' or goal == 'tm':
            grad = -tf.gradients(self.loss(self.xs_ph, self.ys_ph), self.xs_ph)[0]
        elif goal == 'ut':
            grad = tf.gradients(self.loss(self.xs_ph, self.ys_ph), self.xs_ph)[0]
        else:
            raise NotImplementedError
        # flatten the gradient to a 2-D tensor
        # grad_flatten.shape == (batch_size, D)
        grad_flatten = tf.reshape(grad, (batch_size, -1))
        # calculate bound for adversarial noise with the [x_min, x_max] constraint
        delta_xs_min = tf.reshape(self.model.x_min - self.xs_var, (batch_size, -1))
        delta_xs_max = tf.reshape(self.model.x_max - self.xs_var, (batch_size, -1))
        # calculate the adversarial noise for one iteration
        if distance_metric == 'l_2':
            # grad_unit.shape == (batch_size, D)
            grad_unit = get_unit(grad_flatten)
            next_xs_delta = self.delta_xs_var + alpha * grad_unit
            # clip by magnitude
            next_xs_delta = tf.clip_by_norm(next_xs_delta, eps, axes=[1])
        elif distance_metric == 'l_inf':
            # grad_sign.shape == (batch_size, D)
            grad_sign = tf.sign(grad_flatten)
            update = alpha * grad_sign
        else:
            raise NotImplementedError
        # clip by eps

    def config(self, **kwargs):
        pass

    def batch_attack(self, xs, ys=None, ys_target=None):
        pass
