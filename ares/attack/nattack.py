import numpy as np
import tensorflow as tf

from ares.attack.base import Attack
from ares.attack.utils import ConfigVar, Expectation, clip_eta_batch, clip_eta, image_resize, scale


class NAttack(Attack):
    ''' NAttack. A black-box constraint-based method. It is motivated by NES.

    - Supported distance metric: ``l_2``, ``l_inf``.
    - Supported goal: ``t``, ``tm``, ``ut``.
    - References: https://arxiv.org/abs/1905.00441.
    '''

    def __init__(self, model, loss, goal, distance_metric, session, samples_per_draw,
                 samples_batch_size=None, init_distortion=0.001, dimension_reduction=None):
        ''' Initialize NAttack.

        :param model: The model to attack. A ``ares.model.Classifier`` instance.
        :param loss: The loss function to optimize. A ``ares.loss.Loss`` instance.
        :param goal: Adversarial goals. All supported values are ``'t'``, ``'tm'``, and ``'ut'``.
        :param distance_metric: Adversarial distance metric. All supported values are ``'l_2'`` and ``'l_inf'``.
        :param session: The ``tf.Session`` to run the attack in. The ``model`` should be loaded into this session.
        :param samples_per_draw: Number of points to sample for each gradient estimation.
        :param samples_batch_size: Batch size for sampling.
        :param init_distortion: Initial distortion for the Gaussian distribution.
        :param dimension_reduction: ``(height, width)``.
        '''
        self.model, self._session = model, session
        self.goal, self.distance_metric = goal, distance_metric

        # when samples_batch_size is None, set it to samples_per_draw
        self.samples_batch_size = samples_batch_size if samples_batch_size else samples_per_draw
        # for efficiency, samples_per_draw should be multiple of samples_batch_size
        self.samples_per_draw = (samples_per_draw // self.samples_batch_size) * self.samples_batch_size
        self._samples_iteration = self.samples_per_draw // self.samples_batch_size

        self.x_var = tf.Variable(tf.zeros(dtype=self.model.x_dtype, shape=self.model.x_shape))
        self.ys_var = tf.Variable(tf.zeros(dtype=self.model.y_dtype, shape=self.samples_batch_size))

        self.x_ph = tf.placeholder(model.x_dtype, self.model.x_shape)
        self.ys_ph = tf.placeholder(model.y_dtype, (self.samples_batch_size,))

        self.eps = ConfigVar(shape=None, dtype=self.model.x_dtype)
        self.sigma = ConfigVar(shape=None, dtype=tf.float32)
        self.lr = ConfigVar(shape=None, dtype=tf.float32)
        # shape after dimension reduction
        if dimension_reduction:
            assert len(self.model.x_shape) == 3
            perts_shape = (*dimension_reduction, self.model.x_shape[2])
        else:
            perts_shape = self.model.x_shape
        # the mu parameter for the Gaussian
        self.mu_var = tf.Variable(tf.zeros(dtype=self.model.x_dtype, shape=perts_shape))
        mu_var_resized = image_resize(self.mu_var, *self.model.x_shape[:2])
        # sample perturbations from a normal distribution
        perts = tf.random.normal(shape=(self.samples_batch_size, *perts_shape), dtype=self.model.x_dtype)
        mu_perts = self.mu_var + self.sigma.var * perts
        if dimension_reduction:
            mu_perts = image_resize(mu_perts, *self.model.x_shape[:2])
        # linear transform the model's input to range (-1.0, 1.0), and apply arctanh
        arctanh_x = tf.atanh(self._scale_to_tanh(self.x_var))
        # points in the arctanh space to eval loss
        arctanh_points = arctanh_x + mu_perts
        deltas = self._scale_to_model(tf.tanh(arctanh_points)) - self.x_var
        # clip the deltas by magnitude
        clipped_deltas = clip_eta_batch(deltas, self.eps.var, self.distance_metric)
        # points to eval loss
        points = self.x_var + clipped_deltas
        losses = loss(points, self.ys_var)
        perts_mean = tf.reduce_mean(perts, axis=0)
        losses_perts_mean = tf.reduce_mean(tf.reshape(losses, (-1,) + (1,) * len(self.model.x_shape)) * perts, axis=0)
        self.E_perts = Expectation(perts_mean, self._samples_iteration)
        self.E_losses_perts = Expectation(losses_perts_mean, self._samples_iteration)
        self.E_loss_mean = Expectation(tf.reduce_mean(losses), self._samples_iteration)
        self.E_loss_variance = Expectation(tf.math.reduce_std(losses) ** 2, self._samples_iteration)
        loss_std = self.E_loss_variance.val ** 0.5
        loss_mean = self.E_loss_mean.val

        grad = (self.E_losses_perts.val - loss_mean * self.E_perts.val) / ((loss_std + 1e-7) * self.sigma.var)
        if self.goal != 'ut':
            grad = -grad

        # update mu
        self.update_mu_step = self.mu_var.assign_add(self.lr.var * grad)
        delta = self._scale_to_model(tf.tanh(arctanh_x + mu_var_resized)) - self.x_var
        # clip the delta by magnitude
        clipped_delta = clip_eta(delta, self.eps.var, self.distance_metric)
        self.x_adv = self.x_var + clipped_delta

        self.label_pred = self.model.labels(tf.reshape(self.x_adv, (1, *self.model.x_shape)))[0]

        self.setup_mu_step = [self.mu_var.assign(tf.random.normal(perts_shape, stddev=init_distortion))]
        self.setup_x_step = self.x_var.assign(self.x_ph)
        self.setup_ys_step = self.ys_var.assign(self.ys_ph)

        self.logger = None
        self.details = {}

    def config(self, **kwargs):
        ''' (Re)config the attack.

        :param magnitude: Max distortion, should be a float number.
        :param max_queries: Max number of queries, should be an integer.
        :param sigma: Sampling variance (perturbation size) in gradient estimation, should be a float number.
        :param lr: Learning rate (step size) for each iteration, should be a float number.
        :param logger: A standard logger for logging verbose information during attacking.
        '''
        if 'magnitude' in kwargs:
            self._session.run(self.eps.assign, feed_dict={self.eps.ph: kwargs['magnitude']})
        if 'max_queries' in kwargs:
            self.max_queries = kwargs['max_queries']
        if 'sigma' in kwargs:
            self._session.run(self.sigma.assign, feed_dict={self.sigma.ph: kwargs['sigma']})
        if 'lr' in kwargs:
            self._session.run(self.lr.assign, feed_dict={self.lr.ph: kwargs['lr']})
        if 'logger' in kwargs:
            self.logger = kwargs['logger']

    def attack(self, x, y=None, y_target=None):
        ''' Attack one example.

        :return: The generated adversarial example.
        '''
        self._session.run(self.setup_mu_step)
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

        self.details['success'] = False
        queries = 0
        while queries + self.samples_per_draw <= self.max_queries:
            queries += self.samples_per_draw

            self._session.run((self.E_perts.reset, self.E_losses_perts.reset,
                               self.E_loss_mean.reset, self.E_loss_variance.reset))
            for _ in range(self._samples_iteration):
                self._session.run((self.E_perts.update, self.E_losses_perts.update,
                                   self.E_loss_mean.update, self.E_loss_variance.update))

            loss = self._session.run(self.E_loss_mean.val)
            self._session.run(self.update_mu_step)

            if self.logger:
                lr, x_adv_label, x_adv = self._session.run((self.lr.var, self.label_pred, self.x_adv))
                distortion = np.linalg.norm(x_adv - x) if self.distance_metric == 'l_2' else np.max(np.abs(x_adv - x))
                self.logger.info('queries:{}, loss:{}, learning rate:{}, prediction:{}, distortion:{}'.format(
                    queries, loss, lr, x_adv_label, distortion
                ))

            if self._is_adversarial(y, y_target):
                self.details['success'] = True
                break

        self.details['queries'] = queries
        return self._session.run(self.x_adv)

    def _scale_to_model(self, x):
        return scale(x, self.model.x_min, self.model.x_max, -1.0, 1.0)

    def _scale_to_tanh(self, x):
        # np.arctanh(np.tanh(np.arctanh(1.0 - 1e-6) + 10.0)) == 17.242754385535303
        bound = 1.0 - 1e-6
        return scale(x, -bound, bound, self.model.x_min, self.model.x_max)

    def _is_adversarial(self, y, y_target):
        # label of x_adv
        label = self._session.run(self.label_pred)
        if self.goal == 'ut' or self.goal == 'tm':
            return label != y
        else:
            return label == y_target
