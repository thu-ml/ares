import tensorflow as tf

from realsafe.attacks.base import BatchAttack
from realsafe.attacks.utils import get_xs_ph, get_ys_ph, maybe_to_array, get_unit


class FGSM(BatchAttack):
    """
    TODO
    """

    def __init__(self, model, batch_size, loss, goal, distance_metric, session):
        self.model, self.batch_size, self._session = model, batch_size, session

        self.loss, self.goal = loss, goal
        self.distance_metric = distance_metric

        self.xs_ph = get_xs_ph(model, batch_size)
        self.ys_ph = get_ys_ph(model, batch_size)
        self.eps_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.eps_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))

        if goal == 't' or goal == 'tm':
            grad = -tf.gradients(self.loss(self.xs_ph, self.ys_ph), self.xs_ph)[0]
        elif goal == 'ut':
            grad = tf.gradients(self.loss(self.xs_ph, self.ys_ph), self.xs_ph)[0]
        else:
            raise NotImplementedError

        grad_flatten = tf.reshape(grad, (batch_size, -1))

        if distance_metric == 'l_2':
            grad_unit = get_unit(grad_flatten)
            update = tf.expand_dims(self.eps_var, 1) * grad_unit
        elif distance_metric == 'l_inf':
            update = tf.expand_dims(self.eps_var, 1) * tf.sign(grad_flatten)
        else:
            raise NotImplementedError
        update = tf.reshape(update, (self.batch_size, *self.model.x_shape))
        self.xs_adv = tf.clip_by_value(self.xs_ph + update, self.model.x_min, self.model.x_max)

        self.config_setup = self.eps_var.assign(self.eps_ph)

    def config(self, **kwargs):
        eps = maybe_to_array(kwargs['magnitude'], self.batch_size)
        self._session.run(self.config_setup, feed_dict={self.eps_ph: eps})

    def batch_attack(self, xs, ys=None, ys_target=None):
        ls = ys if self.goal == 'ut' else ys_target

        return self._session.run(self.xs_adv, feed_dict={
            self.xs_ph: xs,
            self.ys_ph: ls
        })
