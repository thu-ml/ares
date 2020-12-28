import numpy as np
import tensorflow as tf

from ares.attack.base import Attack
from ares.attack.utils import ConfigVar, Expectation, clip_eta, image_resize


class SPSA(Attack):
    ''' Simultaneous Perturbation Stochastic Approximation (SPSA). A black-box constraint-based method. Use SPSA as
    gradient estimation technique and employ Adam with this estimated gradient to generate the adversarial example.

    - Supported distance metric: ``l_2``, ``l_inf``.
    - Supported goal: ``t``, ``tm``, ``ut``.
    - References: https://arxiv.org/abs/1802.05666.
    '''

    def __init__(self, model, loss, goal, distance_metric, session, samples_per_draw, samples_batch_size=None,
                 dimension_reduction=None):
        ''' Initialize SPSA.

        :param model: The model to attack. A ``ares.model.Classifier`` instance.
        :param loss: The loss function to optimize. A ``ares.loss.Loss`` instance.
        :param goal: Adversarial goals. All supported values are ``'t'``, ``'tm'``, and ``'ut'``.
        :param distance_metric: Adversarial distance metric. All supported values are 'l_2' and 'l_inf'.
        :param session: The ``tf.Session`` to run the attack in. The ``model`` should be loaded into this session.
        :param samples_per_draw: Number of points to sample for each gradient estimation.
        :param samples_batch_size: Batch size for sampling.
        :param dimension_reduction: ``(height, width)``.
        '''
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

        self.label_pred = self.model.labels(tf.reshape(self.x_adv_var, (1, *self.model.x_shape)))[0]

        # perturbations
        if dimension_reduction:
            assert len(self.model.x_shape) == 3
            perts_shape = (self.samples_batch_size // 2, *dimension_reduction, self.model.x_shape[2])
            perts = tf.random.uniform(perts_shape, minval=-1.0, maxval=1.0, dtype=self.model.x_dtype)
            perts = image_resize(perts, *self.model.x_shape[:2])
            perts = tf.sign(perts)
            perts = tf.concat([perts, tf.negative(perts)], axis=0)
        else:
            perts_shape = (self.samples_batch_size // 2, *self.model.x_shape)
            perts = tf.sign(tf.random.uniform(perts_shape, minval=-1.0, maxval=1.0, dtype=self.model.x_dtype))
            perts = tf.concat([perts, tf.negative(perts)], axis=0)
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

        # Adam
        self.beta1, self.beta1_init = ConfigVar(None, self.model.x_dtype), 0.9
        self.beta2, self.beta2_init = ConfigVar(None, self.model.x_dtype), 0.999
        self.epsilon, self.epsilon_init = ConfigVar(None, self.model.x_dtype), 1e-9
        self.m_var = tf.Variable(tf.zeros(self.model.x_shape, self.model.x_dtype))
        self.v_var = tf.Variable(tf.zeros(self.model.x_shape, self.model.x_dtype))
        self.t_var = tf.Variable(0.0, dtype=self.model.x_dtype)
        self.prepare_adam_step = (
            tf.variables_initializer([self.m_var, self.v_var, self.t_var]),
            self.beta1.assign, self.beta2.assign, self.epsilon.assign,
        )
        self.adam_step = (
            self.t_var.assign_add(1.0),
            self.m_var.assign(self.beta1.var * self.m_var + (1.0 - self.beta1.var) * grad),
            self.v_var.assign(self.beta2.var * self.v_var + (1.0 - self.beta2.var) * grad * grad),
        )
        m_hat = self.m_var / (1.0 - tf.pow(self.beta1.var, self.t_var))
        v_hat = self.v_var / (1.0 - tf.pow(self.beta2.var, self.t_var))
        # update the adversarial example
        x_adv_delta = self.x_adv_var - self.x_var + self.lr.var * m_hat / (tf.sqrt(v_hat) + self.epsilon.var)
        x_adv_next = self.x_var + clip_eta(x_adv_delta, self.eps.var, self.distance_metric)
        x_adv_next = tf.clip_by_value(x_adv_next, self.model.x_min, self.model.x_max)
        self.update_x_adv_step = self.x_adv_var.assign(x_adv_next)

        self.setup_x_step = [self.x_var.assign(self.x_ph), self.x_adv_var.assign(self.x_ph)]
        self.setup_ys_step = self.ys_var.assign(self.ys_ph)

        self.logger = None
        self.details = {}

    def config(self, **kwargs):
        ''' (Re)config the attack.

        :param magnitude: Max distortion, should be a float number.
        :param max_queries: Max number of queries, should be an integer.
        :param sigma: Sampling variance (perturbation size) in gradient estimation, should be a float number.
        :param lr: Learning rate of Adam optimizer, should be a float number.
        :param beta1: First-order momentum of Adam optimizer, should be a float number.
        :param beta2: Second-order momentum of Adam optimizer, should be a float number.
        :param epsilon: A small float number to prevent division by zero in Adam.
        :param logger: A standard logger for logging verbose information during attack.
        '''
        if 'magnitude' in kwargs:
            self._session.run(self.eps.assign, feed_dict={self.eps.ph: kwargs['magnitude']})
        if 'max_queries' in kwargs:
            self.max_queries = kwargs['max_queries']
        if 'sigma' in kwargs:
            self._session.run(self.sigma.assign, feed_dict={self.sigma.ph: kwargs['sigma']})
        if 'lr' in kwargs:
            self._session.run(self.lr.assign, feed_dict={self.lr.ph: kwargs['lr']})
        if 'beta1' in kwargs:
            self.beta1_init = kwargs['beta1']
        if 'beta2' in kwargs:
            self.beta2_init = kwargs['beta2']
        if 'epsilon' in kwargs:
            self.epsilon_init = kwargs['epsilon']
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
        ''' Attack one example.

        :return: The generated adversarial example.
        '''
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

        self._session.run(self.prepare_adam_step, feed_dict={
            self.beta1.ph: self.beta1_init,
            self.beta2.ph: self.beta2_init,
            self.epsilon.ph: self.epsilon_init,
        })

        self.details['success'] = False
        queries = 0
        while queries + self.samples_per_draw <= self.max_queries:
            queries += self.samples_per_draw

            self._session.run((self.E_grad.reset, self.E_mean_loss.reset))
            for _ in range(self._samples_iteration):
                self._session.run((self.E_grad.update, self.E_mean_loss.update))

            self._session.run(self.adam_step)
            loss, _ = self._session.run((self.E_mean_loss.val, self.update_x_adv_step))

            if self.logger:
                lr, x_adv_label, x_adv = self._session.run((self.lr.var, self.label_pred, self.x_adv_var))
                distortion = np.linalg.norm(x_adv - x) if self.distance_metric == 'l_2' else np.max(np.abs(x_adv - x))
                self.logger.info('queries:{}, loss:{}, learning rate:{}, prediction:{}, distortion:{}'.format(
                    queries, np.mean(loss), lr, x_adv_label, distortion
                ))

            if self._is_adversarial(y, y_target):
                self.details['success'] = True
                break

        self.details['queries'] = queries
        return self._session.run(self.x_adv_var)
