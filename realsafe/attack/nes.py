import numpy as np
import tensorflow as tf

from realsafe.attack.base import Attack
from realsafe.attack.utils import ConfigVar, Expectation


class NES(Attack):
    '''
    Natural Evolution Strategies (NES)
    A black-box constraint-based method. Use NES as gradient estimation technique and employ PGD with this estimated
    gradient to generate the adversarial example.

    Supported distance metric: `l_2`, `l_inf`
    Supported goal: `t`, `tm`, `ut`
    Supported config parameters:
    - `magnitude`: max distortion, should be a float number.
    - `max_queries`: max number of queries, should be a integer.
    - `sigma`: sampling variance (perturbation size) in gradient estimation, should be a float number.
    - `lr`: learning rate (step size) for each iteration, should be a float number.
    - `min_lr`: min learning rate if `lr_tuning=True`, should be a float number.
    - `lr_tuning`: a bool, represents whether or not to decay the learning rate if the loss plateaus.
    - `plateau_length`: an integer, represents the number of iterations when the loss plateaus to decay learning rate.
    - `logger`: a standard logger for logging verbose information during attack.

    References:
    [1] https://arxiv.org/abs/1804.08598
    [2] http://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
    '''

    def __init__(self, model, loss, goal, distance_metric, session, samples_per_draw, samples_batch_size=None):
        self.model, self._session = model, session
        self.goal, self.distance_metric = goal, distance_metric

        # when samples_batch_size is None, set it to samples_per_draw
        samples_batch_size = samples_batch_size if samples_batch_size else samples_per_draw
        self.samples_batch_size = (samples_batch_size // 2) * 2
        # for efficiency, samples_per_draw should be multiple of samples_batch_size
        self.samples_per_draw = (samples_per_draw // self.samples_batch_size) * self.samples_batch_size
        self._samples_iteration = self.samples_per_draw // self.samples_batch_size

        self.x_var = tf.Variable(tf.zeros(dtype=self.model.x_dtype, shape=self.model.x_shape))
        self.x_adv_var = tf.Variable(tf.zeros(dtype=self.model.x_dtype, shape=self.model.x_shape))
        self.ys_var = tf.Variable(tf.zeros(dtype=self.model.y_dtype, shape=self.samples_batch_size))

        self.x_ph = tf.placeholder(model.x_dtype, self.model.x_shape)
        self.ys_ph = tf.placeholder(model.y_dtype, (self.samples_batch_size,))

        self.eps = ConfigVar(shape=None, dtype=self.model.x_dtype)
        self.sigma = ConfigVar(shape=None, dtype=tf.float32)
        self.lr = ConfigVar(shape=None, dtype=tf.float32)

        self.label_pred = self.model.logits_and_labels(tf.reshape(self.x_adv_var, (1, *self.model.x_shape)))[1][0]

        # pertubations
        perts = tf.random.normal(shape=(self.samples_batch_size // 2, *self.model.x_shape), dtype=self.model.x_dtype)
        perts = tf.concat([perts, -perts], axis=0)
        # points to eval loss
        points = self.x_adv_var + self.sigma.var * perts
        loss = loss(points, self.ys_var)
        # estimated gradient
        grads = tf.reshape(loss, [-1] + [1] * len(self.model.x_shape)) * perts
        grad = tf.reduce_mean(grads, axis=0) / self.sigma.var
        self.E_grad = Expectation(grad, self._samples_iteration)
        self.E_mean_loss = Expectation(tf.reduce_mean(loss), self._samples_iteration)

        grad = self.E_grad.val
        if self.goal != 'ut':
            grad = -grad

        # update the adversarial example
        if self.distance_metric == 'l_2':
            grad_norm = tf.maximum(1e-12, tf.norm(grad))
            x_adv_delta = self.x_adv_var - self.x_var + self.lr.var * grad / grad_norm
            x_adv_next = self.x_var + tf.clip_by_norm(x_adv_delta, self.eps.var)
        elif self.distance_metric == 'l_inf':
            x_adv_delta = self.x_adv_var - self.x_var + self.lr.var * tf.sign(grad)
            x_adv_next = self.x_var + tf.clip_by_value(x_adv_delta, tf.negative(self.eps.var), self.eps.var)
        else:
            raise NotImplementedError
        x_adv_next = tf.clip_by_value(x_adv_next, self.model.x_min, self.model.x_max)
        self.update_x_adv_step = self.x_adv_var.assign(x_adv_next)

        self.setup_x_step = [self.x_var.assign(self.x_ph), self.x_adv_var.assign(self.x_ph)]
        self.setup_ys_step = self.ys_var.assign(self.ys_ph)

        self.logger = None
        self.details = {}

    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self._session.run(self.eps.assign, feed_dict={self.eps.ph: kwargs['magnitude']})
        if 'max_queries' in kwargs:
            self.max_queries = kwargs['max_queries']
        if 'sigma' in kwargs:
            self._session.run(self.sigma.assign, feed_dict={self.sigma.ph: kwargs['sigma']})
        if 'lr' in kwargs:
            self.init_lr = kwargs['lr']
        if 'min_lr' in kwargs:
            self.min_lr = kwargs['min_lr']
        if 'lr_tuning' in kwargs:
            self.lr_tuning = kwargs['lr_tuning']
        if 'plateau_length' in kwargs:
            self.plateau_length = kwargs['plateau_length']
        if 'logger' in kwargs:
            self.logger = kwargs['logger']

    def _is_adversarial(self, y, y_target):
        # label of x_adv
        label = self._session.run(self.label_pred)
        if self.goal == 'ut' or self.goal == 'tm':
            return label != y
        else:
            return label == y_target

    def attack(self, x, y=None, y_target=None):
        self._session.run(self.setup_x_step, feed_dict={self.x_ph: x})
        self._session.run(self.setup_ys_step, feed_dict={
            self.ys_ph: np.repeat(y if self.goal == 'ut' else y_target, self.samples_batch_size)
        })

        if self._is_adversarial(y, y_target):
            if self.logger:
                self.logger.info('Original image is already adversarial')
            self.details['queries'] = 0
            self.details['success'] = True
            return x

        last_loss = []
        lr = self.init_lr
        self._session.run(self.lr.assign, feed_dict={self.lr.ph: lr})
        
        self.details['success'] = False
        queries = 0
        while queries + self.samples_per_draw <= self.max_queries:
            queries += self.samples_per_draw

            self._session.run((self.E_grad.reset, self.E_mean_loss.reset))
            for _ in range(self._samples_iteration):
                self._session.run((self.E_grad.update, self.E_mean_loss.update))
            loss, _ = self._session.run((self.E_mean_loss.val, self.update_x_adv_step))

            if self.lr_tuning:
                last_loss.append(np.mean(loss))
                last_loss = last_loss[-self.plateau_length:]
                if len(last_loss) == self.plateau_length:
                    if self.goal == 'ut' and last_loss[-1] < last_loss[0]:
                        lr = max(lr / 2, self.min_lr)
                        self._session.run(self.lr.assign, feed_dict={self.lr.ph: lr})
                        last_loss = []
                    elif self.goal != 'ut' and last_loss[-1] > last_loss[0]:
                        lr = max(lr / 2, self.min_lr)
                        self._session.run(self.lr.assign, feed_dict={self.lr.ph: lr})
                        last_loss = []

            if self.logger:
                x_adv_label, x_adv = self._session.run((self.label_pred, self.x_adv_var))
                distortion = np.linalg.norm(x_adv - x) if self.distance_metric == 'l_2' else np.max(np.abs(x_adv - x))
                self.logger.info('queries:{}, loss:{}, learning rate:{}, prediction:{}, distortion:{}'.format(
                    queries, np.mean(loss), lr, x_adv_label, distortion
                ))

            if self._is_adversarial(y, y_target):
                self.details['success'] = True
                break

        self.details['queries'] = queries
        return self._session.run(self.x_adv_var)
