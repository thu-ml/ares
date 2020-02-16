import tensorflow as tf
import numpy as np
import collections

from realsafe.attack.base import BatchAttack
from realsafe.attack.utils import mean_square_distance


class Boundary(BatchAttack):
    '''
    Boundary
    A black-box decision-based method.

    Supported distance metric: `l_2`
    Supported goal: `t`, `tm`, `ut`
    Supported config parameters:
    - `magnitude`: max distortion, could be either a float number or a numpy float number array with shape of

    References:
    [1]
    '''

    def __init__(self, model, batch_size, goal, session):
        self.model, self.batch_size, self.goal, self._session = model, batch_size, goal, session

        self.xs_ph = tf.placeholder(model.x_dtype, (self.batch_size, *self.model.x_shape))
        self.labels = self.model.labels(self.xs_ph)

        self.logger = None

    def config(self, **kwargs):
        if 'starting_points' in kwargs:
            self.starting_points = kwargs['starting_points']

        if 'max_queries' in kwargs:
            self.max_queries = kwargs['max_queries']
        if 'max_directions' in kwargs:
            self.max_directions = kwargs['max_directions']

        if 'spherical_step' in kwargs:
            self.spherical_step = kwargs['spherical_step']
        if 'source_step' in kwargs:
            self.source_step = kwargs['source_step']
        if 'step_adaptation' in kwargs:
            self.step_adaptation = kwargs['step_adaptation']

        if 'logger' in kwargs:
            self.logger = kwargs['logger']

    def _is_adversarial(self, label, y, y_target):
        if self.goal == 'ut' or self.goal == 'tm':
            return label != y
        else:
            return label == y_target

    def _attack_one(self, index):
        x = self._xs_adv[index].copy()
        y = None if self._ys is None else self._ys[index]
        y_target = None if self._ys_target is None else self._ys_target[index]

        if self._is_adversarial(self._preds[index], y, y_target):
            if self.logger:
                self.logger.info('The {}-th original image is already adversarial'.format(index))
            yield
            for _ in range(self.max_queries):
                yield
            return

        x_min, x_max = self.model.x_min, self.model.x_max
        x_shape, x_dtype = self.model.x_shape, self.model.x_dtype.as_numpy_dtype

        spherical_step, source_step, step_adaptation = self.spherical_step, self.source_step, self.step_adaptation

        x_adv = self.starting_points[index]
        dist = mean_square_distance(x_adv, x, x_min, x_max)
        dist_per_query = np.zeros([self.max_queries + 1])
        stats_spherical_adversarial = collections.deque(maxlen=100)
        stats_step_adversarial = collections.deque(maxlen=30)

        self._xs_adv[index] = x_adv
        yield
        x_adv_label = self._preds[index]

        if self.logger:
            self.logger.info('step {}: index={}, {:.5e}, prediction={}, stepsizes={:.1e}/{:.1e}: {}'.format(
                0, index, x_adv_label, dist, spherical_step, source_step, ''
            ))
        dist_per_query[0] = dist

        step, queries, last_queries = 0, 0, 0
        while True:
            step += 1

            unnormalized_source_direction = x - x_adv
            source_norm = np.linalg.norm(unnormalized_source_direction)
            source_direction = unnormalized_source_direction / source_norm

            do_spherical = (step % 10 == 0)

            for _ in range(self.max_directions):
                perturbation = np.random.normal(0.0, 1.0, x_shape).astype(x_dtype)
                dot = np.vdot(perturbation, source_direction)
                perturbation -= dot * source_direction
                perturbation *= spherical_step * source_norm / np.linalg.norm(perturbation)

                D = 1 / np.sqrt(spherical_step ** 2.0 + 1)
                direction = perturbation - unnormalized_source_direction
                spherical_candidate = np.clip(x + D * direction, x_min, x_max)

                new_source_direction = x - spherical_candidate
                new_source_direction_norm = np.linalg.norm(new_source_direction)
                length = source_step * source_norm

                deviation = new_source_direction_norm - source_norm
                length = max(0, length + deviation) / new_source_direction_norm
                candidate = np.clip(spherical_candidate + length * new_source_direction, x_min, x_max)

                if do_spherical:
                    self._xs_adv[index] = spherical_candidate
                    yield
                    spherical_is_adversarial = self._is_adversarial(self._preds[index], y, y_target)
                    queries += 1
                    if queries == self.max_queries:
                        self._xs_adv[index] = x_adv
                        return
                    stats_spherical_adversarial.appendleft(spherical_is_adversarial)

                    if not spherical_is_adversarial:
                        continue

                self._xs_adv[index] = candidate
                yield
                is_adversarial = self._is_adversarial(self._preds[index], y, y_target)
                queries += 1
                if queries == self.max_queries:
                    self._xs_adv[index] = x_adv
                    return
                if do_spherical:
                    stats_step_adversarial.appendleft(is_adversarial)

                if not is_adversarial:
                    continue

                new_x_adv = candidate
                new_dist = mean_square_distance(new_x_adv, x, x_min, x_max)
                break
            else:
                new_x_adv = None

            dist_per_query[last_queries:min(queries, self.max_queries + 1)] = dist

            message = ''
            if new_x_adv is not None:
                abs_improvement = dist - new_dist
                rel_improvement = abs_improvement / dist
                message = 'd. reduced by {:.2f}% ({:.4e})'.format(rel_improvement * 100, abs_improvement)
                x_adv_label, x_adv, dist = self._preds[index], new_x_adv, new_dist

            if self.logger:
                self.logger.info('step {}: index={}, {:.5e}, prediction={}, stepsizes={:.1e}/{:.1e}: {}'.format(
                    0, index, x_adv_label, dist, spherical_step, source_step, message
                ))

            if len(stats_step_adversarial) == stats_step_adversarial.maxlen and \
                    len(stats_spherical_adversarial) == stats_spherical_adversarial.maxlen:
                p_spherical = np.mean(stats_spherical_adversarial)
                p_step = np.mean(stats_step_adversarial)
                n_spherical = len(stats_spherical_adversarial)
                n_step = len(stats_step_adversarial)

                if p_spherical > 0.5:
                    message = 'Boundary too linear, increasing steps:'
                    spherical_step *= step_adaptation
                    source_step *= step_adaptation
                elif p_spherical < 0.2:
                    message = 'Boundary too non-linear, decreasing steps:'
                    spherical_step /= step_adaptation
                    source_step /= step_adaptation
                else:
                    message = None

                if message is not None:
                    stats_spherical_adversarial.clear()
                    if self.logger:
                        self.logger.info(' {} {:.2f} ({:3d}), {:.2f} ({:3d})'.format(
                            message, p_spherical, n_spherical, p_step, n_step
                        ))

                if p_step > 0.5:
                    message = 'Success rate too high, increasing source step:'
                    source_step *= step_adaptation
                elif p_step < 0.2:
                    message = 'Success rate too low, decreasing source step:'
                    source_step /= step_adaptation
                else:
                    message = None

                if message is not None:
                    stats_step_adversarial.clear()
                    if self.logger:
                        self.logger.info(' {} {:.2f} ({:3d}), {:.2f} ({:3d})'.format(
                            message, p_spherical, n_spherical, p_step, n_step
                        ))

            last_queries = queries
            if queries == self.max_queries:
                self._xs_adv[index] = x_adv
                return

    def _run_attackers(self):
        self._preds = self._session.run(self.labels, feed_dict={self.xs_ph: self._xs_adv})
        for i in range(self.batch_size):
            try:
                next(self._attackers[i])
            except StopIteration:
                pass

    def batch_attack(self, xs, ys=None, ys_target=None):
        self._xs_adv, self._ys, self._ys_target = xs.copy(), ys, ys_target
        self._attackers = [self._attack_one(i) for i in range(self.batch_size)]

        self._run_attackers()
        self._run_attackers()

        for _ in range(self.max_queries):
            self._run_attackers()

        xs_adv = self._xs_adv
        delattr(self, '_attackers')
        delattr(self, '_preds')
        delattr(self, '_xs_adv')
        delattr(self, '_ys')
        delattr(self, '_ys_target')

        return xs_adv
