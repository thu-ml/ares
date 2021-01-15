import numpy as np
import tensorflow as tf
from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph
from ares.loss import CrossEntropyLoss


class Attacker(BatchAttack):
    def __init__(self, model, batch_size, dataset, session):
        """ Based on ares.attack.bim.BIM, numpy version. """
        self.model, self.batch_size, self._session = model, batch_size, session
        # dataset == "imagenet" or "cifar10"
        loss = CrossEntropyLoss(self.model)
        # placeholder for batch_attack's input
        self.xs_ph = get_xs_ph(model, batch_size)
        self.ys_ph = get_ys_ph(model, batch_size)
        self.loss = loss(self.xs_ph, self.ys_ph)
        self.grad = tf.gradients(self.loss, self.xs_ph)[0]
        self.iteration = 10

    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self.eps = kwargs['magnitude'] - 1e-6
            self.alpha = self.eps / 7

    def batch_attack(self, xs, ys=None, ys_target=None):
        xs_lo, xs_hi = xs - self.eps, xs + self.eps
        xs_adv = xs
        for _ in range(self.iteration):
            grad = self._session.run(self.grad, feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
            grad = grad.reshape(self.batch_size, *self.model.x_shape)
            grad_sign = np.sign(grad)
            xs_adv = np.clip(xs_adv + self.alpha * grad_sign, xs_lo, xs_hi)
            xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)
        return xs_adv
