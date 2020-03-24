import tensorflow as tf
import numpy as np
import multiprocessing as mp
import ctypes
import collections
import cv2

from realsafe.attack.base import BatchAttack
from realsafe.attack.utils import mean_square_distance


class AttackCtx(object):
    def __init__(self, attacker, index, xs_batch_array, ys_batch_array, dist_array,
                 starting_points, ys, ys_target, pipe):
        self.x_dtype, self.y_dtype = attacker.model.x_dtype.as_numpy_dtype, attacker.model.y_dtype.as_numpy_dtype
        self.batch_size, self.x_shape = attacker.batch_size, attacker.model.x_shape
        self.goal = attacker.goal
        self.index = index
        self.x_min, self.x_max = attacker.model.x_min, attacker.model.x_max
        self.y = None if ys is None else ys[index]
        self.y_target = None if ys_target is None else ys_target[index]
        self.mu = attacker.mu
        self.sigma = attacker.sigma
        self.decay_factor = attacker.decay_factor
        self.c = attacker.c
        self.max_queries = attacker.max_queries
        self.pipe = pipe
        self.starting_point = starting_points[index]
        self.dimension_reduction = attacker.dimension_reduction

        self._xs_batch_array, self._ys_batch_array = xs_batch_array, ys_batch_array
        self._dist_array = dist_array

        self.logs = [] if attacker.logger else None

    def _is_adversarial(self, label):
        if self.goal == 'ut' or self.goal == 'tm':
            return label != self.y
        else:
            return label == self.y_target

    def worker(self):
        task = self.run()
        next(task)

        while True:
            self.pipe.recv()
            try:
                next(task)
                self.pipe.send((False, self.logs.copy() if self.logs else []))
                self.logs = [] if self.logs is not None else None
            except StopIteration:
                self.pipe.send((True, self.logs.copy() if self.logs else []))
                return

    def run(self):
        index = self.index
        xs_batch = np.frombuffer(self._xs_batch_array, dtype=self.x_dtype).reshape((self.batch_size, *self.x_shape))
        ys_batch = np.frombuffer(self._ys_batch_array, dtype=self.y_dtype).reshape((self.batch_size,))

        x = xs_batch[index].copy()

        yield
        if self._is_adversarial(ys_batch[index]):
            if self.logs is not None:
                self.logs.append('{}: The original image is already adversarial'.format(index))
            return

        x_adv = self.starting_point
        dist = mean_square_distance(x, x_adv, self.x_min, self.x_max)
        dist_per_query = np.frombuffer(self._dist_array, dtype=np.float).reshape((self.batch_size, -1))
        stats_adversarial = collections.deque(maxlen=30)

        if self.dimension_reduction:
            assert len(self.x_shape) == 3
            pert_shape = (*self.dimension_reduction, self.x_shape[2])
        else:
            pert_shape = self.x_shape

        N = np.prod(pert_shape)
        K = int(N / 20)

        evolution_path = np.zeros(pert_shape, dtype=self.x_dtype)
        diagonal_covariance = np.ones(pert_shape, dtype=self.x_dtype)

        xs_batch[self.index] = x_adv
        yield
        x_adv_label = ys_batch[self.index]

        if self.logs is not None:
            self.logs.append('{}: step {}, {:.5e}, prediction={}, stepsizes={:.1e}/{:.1e}: {}'.format(
                self.index, 0, dist, x_adv_label, self.sigma, self.mu, ''
            ))

        dist_per_query[self.index][0] = dist

        for step in range(1, self.max_queries + 1):
            unnormalized_source_direction = x - x_adv
            source_norm = np.linalg.norm(unnormalized_source_direction)

            selection_probability = diagonal_covariance.reshape(-1) / np.sum(diagonal_covariance)
            selected_indices = np.random.choice(N, K, replace=False, p=selection_probability)

            perturbation = np.random.normal(0.0, 1.0, pert_shape).astype(self.x_dtype)
            factor = np.zeros([N], dtype=self.x_dtype)
            factor[selected_indices] = 1
            perturbation *= factor.reshape(pert_shape) * np.sqrt(diagonal_covariance)

            if self.dimension_reduction:
                perturbation_large = cv2.resize(perturbation, self.x_shape[:2])
            else:
                perturbation_large = perturbation

            biased = x_adv + self.mu * unnormalized_source_direction
            candidate = biased + self.sigma * source_norm * perturbation_large / np.linalg.norm(perturbation_large)
            candidate = x - (x - candidate) / np.linalg.norm(x - candidate) * np.linalg.norm(x - biased)
            candidate = np.clip(candidate, self.x_min, self.x_max)

            xs_batch[self.index] = candidate
            yield
            is_adversarial = self._is_adversarial(ys_batch[self.index])
            stats_adversarial.appendleft(is_adversarial)

            if is_adversarial:
                new_x_adv = candidate
                new_dist = mean_square_distance(new_x_adv, x, self.x_min, self.x_max)
                evolution_path = self.decay_factor * evolution_path + np.sqrt(1 - self.decay_factor ** 2) * perturbation
                diagonal_covariance = (1 - self.c) * diagonal_covariance + self.c * (evolution_path ** 2)
            else:
                new_x_adv = None

            message = ''
            if new_x_adv is not None:
                abs_improvement = dist - new_dist
                rel_improvement = abs_improvement / dist
                message = 'd. reduced by {:.2f}% ({:.4e})'.format(rel_improvement * 100, abs_improvement)

                x_adv, dist = new_x_adv, new_dist
                x_adv_label = ys_batch[self.index]

            dist_per_query[self.index][step] = dist

            if self.logs is not None:
                self.logs.append('{}: step {}, {:.5e}, prediction={}, stepsizes={:.1e}/{:.1e}: {}'.format(
                    self.index, step, dist, x_adv_label, self.sigma, self.mu, message
                ))

            if len(stats_adversarial) == stats_adversarial.maxlen:
                p_step = np.mean(stats_adversarial)
                self.mu *= np.exp(p_step - 0.2)
                stats_adversarial.clear()

        xs_batch[self.index] = x_adv
        return


class Evolutionary(BatchAttack):
    '''
    Evolutionary
    A black-box decision-based method.

    Supported distance metric: `l_2`
    Supported goal: `t`, `tm`, `ut`

    References:
    [1] TODO
    '''

    def __init__(self, model, batch_size, goal, session, dimension_reduction=None):
        self.model, self.batch_size, self.goal, self._session = model, batch_size, goal, session
        self.dimension_reduction = dimension_reduction

        self.xs_ph = tf.placeholder(model.x_dtype, (self.batch_size, *self.model.x_shape))
        self.labels = self.model.labels(self.xs_ph)
        self.details = {}

        self.logger = None

    def config(self, **kwargs):
        '''
        :param starting_points: Starting points which are already adversarial. A numpy array with data type of
            `self.x_dtype`, with shape of `(self.batch_size, *self.x_shape)`.
        :param max_queries: Max queries. An integer.
        :param mu: A hyper-parameter controlling the mean of the Gaussian distribution. A float number.
        :param sigma: A hyper-parameter controlling the variance of the Gaussian distribution. A float number.
        :param decay_factor: The decay factor for the evolution path. A float number.
        :param c: The decay factor for the covariance matrix. A float number.
        :param logger: A standard logger for logging verbose information during attacking.
        '''
        if 'starting_points' in kwargs:
            self.starting_points = kwargs['starting_points']

        if 'max_queries' in kwargs:
            self.max_queries = kwargs['max_queries']

        if 'mu' in kwargs:
            self.mu = kwargs['mu']
        if 'sigma' in kwargs:
            self.sigma = kwargs['sigma']
        if 'decay_factor' in kwargs:
            self.decay_factor = kwargs['decay_factor']
        if 'c' in kwargs:
            self.c = kwargs['c']

        if 'logger' in kwargs:
            self.logger = kwargs['logger']

    def batch_attack(self, xs, ys=None, ys_target=None):
        xs_batch_array = mp.RawArray(ctypes.c_byte, np.array(xs).astype(self.model.x_dtype.as_numpy_dtype).nbytes)
        xs_shape = (self.batch_size, *self.model.x_shape)
        xs_batch = np.frombuffer(xs_batch_array, dtype=self.model.x_dtype.as_numpy_dtype).reshape(xs_shape)
        xs_batch[:] = xs
        self.details['dist_per_query'] = np.zeros((self.batch_size, self.max_queries + 1), dtype=np.float)
        dist_array = mp.RawArray(ctypes.c_byte, self.details['dist_per_query'].nbytes)

        lbs = ys_target if ys is None else ys
        ys_batch_array = mp.RawArray(ctypes.c_byte, np.array(lbs).astype(self.model.y_dtype.as_numpy_dtype).nbytes)
        ys_batch_shape = (self.batch_size,)
        ys_batch = np.frombuffer(ys_batch_array, dtype=self.model.y_dtype.as_numpy_dtype).reshape(ys_batch_shape)

        workers = []
        for index in range(self.batch_size):
            local, remote = mp.Pipe()
            ctx = AttackCtx(self, index, xs_batch_array, ys_batch_array, dist_array,
                            self.starting_points, ys, ys_target, remote)
            worker = mp.Process(target=AttackCtx.worker, args=(ctx,))
            worker.start()
            workers.append((worker, local))

        i = 0
        while True:
            i += 1
            ys_batch[:] = self._session.run(self.labels, feed_dict={self.xs_ph: xs_batch})
            for _, pipe in workers:
                if not pipe.closed:
                    pipe.send(())
            flag = False
            for _, pipe in workers:
                if not pipe.closed:
                    stop, logs = pipe.recv()
                    if self.logger:
                        for log in logs:
                            self.logger.info(log)
                    if stop:
                        pipe.close()
                    else:
                        flag = True
            if not flag:
                break

        dist_per_query = np.frombuffer(dist_array, dtype=np.float).reshape((self.batch_size, -1))
        np.copyto(self.details['dist_per_query'], dist_per_query)
        return xs_batch.copy()
