import numpy as np
import tensorflow as tf

from ares.benchmark.utils import load_attack, gen_starting_points
from ares.dataset import dataset_to_iterator


class IterationBenchmark(object):
    ''' Iteration benchmark. '''

    def __init__(self, iteration, attack_name, model, batch_size, dataset_name, goal, distance_metric, session,
                 cw_n_points=10, **kwargs):
        ''' Initialize IterationBenchmark.

        :param iteration: The iteration count. For bim, pgd, mim, cw, deepfool attack, it would be passed to the attack
            as the ``iteration`` configuration parameter. For nes, spsa, nattack, boundary, evolutionary attack, it
            would be passed to the attack as the ``max_queries`` configuration parameter.
        :param attack_name: The attack method's name. All valid values are ``'bim'``, ``'pgd'``, ``'mim'``, ``'cw'``,
            ``'deepfool'``, ``'nes'``, ``'spsa'``, ``'nattack'``, ``'boundary'``, ``'evolutionary'``.
        :param model: The classifier model to run the attack on.
        :param batch_size: Batch size for attack.
        :param dataset_name: The dataset's name. All valid values are ``'cifar10'`` and ``'imagenet'``.
        :param goal: The adversarial goal for the attack method. All valid values are ``'t'`` for targeted attack,
            ``'tm'`` for targeted missclassification attack, and ``'ut'`` for untargeted attack.
        :param distance_metric: The adversarial distance metric for the attack method. All valid values are ``'l_2'``
            and ``'l_inf'``.
        :param session: The ``tf.Session`` instance for the attack to run in.
        :param cw_n_points: How many times should we run 'cw' attack for the benchmark. To get the benchmark result for
            'cw' attack, we need to run it for each iteration parameter we are interested in. Since the computation cost
            for C&W attack is huge, we select ``cw_n_points`` numbers between 0 and ``iteration`` uniformly as the
            iteration parameter to reduce the computation cost.
        :param kwargs: Other keyword arguments to pass to the attack method's initialization function.
        '''
        self.iteration = iteration

        def iteration_callback(xs, xs_adv):
            labels = model.labels(xs_adv)

            delta_xs = tf.reshape(xs_adv - xs, (xs.shape[0], -1))
            if distance_metric == 'l_2':
                dists = tf.linalg.norm(delta_xs, axis=1)
            else:  # 'l_inf'
                dists = tf.reduce_max(tf.abs(delta_xs), axis=1)

            return labels, dists

        init_kwargs = dict()
        init_kwargs['model'] = model
        init_kwargs['batch_size'] = batch_size
        init_kwargs['samples_batch_size'] = batch_size  # use same batch_size for nes, spsa, nattack's samples
        init_kwargs['goal'] = goal
        init_kwargs['distance_metric'] = distance_metric
        init_kwargs['session'] = session
        for k, v in kwargs.items():
            init_kwargs[k] = v
        init_kwargs['iteration_callback'] = iteration_callback

        self.model = model
        self.attack_name, self.dataset_name = attack_name, dataset_name
        self.batch_size, self.goal, self.distance_metric = batch_size, goal, distance_metric
        self.attack = load_attack(attack_name, init_kwargs)
        self.cw_n_points = cw_n_points

        self._session = session

        if self.attack_name in ('bim', 'pgd', 'mim'):
            self._run = self._run_basic
        elif self.attack_name == 'cw':
            self._xs_ph = tf.placeholder(self.model.x_dtype, shape=(self.batch_size, *self.model.x_shape))
            self._xs_adv_ph = tf.placeholder(self.model.x_dtype, shape=(self.batch_size, *self.model.x_shape))
            self._cw_data = iteration_callback(self._xs_ph, self._xs_adv_ph)
            self._run = self._run_cw
        elif self.attack_name == 'deepfool':
            self._run = self._run_deepfool
        elif self.attack_name in ('nes', 'spsa', 'nattack'):
            self._x_ph = tf.placeholder(self.model.x_dtype, shape=self.model.x_shape)
            self._x_adv_ph = tf.placeholder(self.model.x_dtype, shape=self.model.x_shape)
            self._score_based_data = iteration_callback(
                tf.reshape(self._x_ph, (1, *self.model.x_shape)),
                tf.reshape(self._x_adv_ph, (1, *self.model.x_shape)),
            )
            self._run = self._run_score_based
        elif self.attack_name in ('boundary', 'evolutionary'):
            self._xs_ph = tf.placeholder(model.x_dtype, shape=(None, *model.x_shape))
            self._xs_label = model.labels(self._xs_ph)
            self._run = self._run_decision_based
        else:
            raise NotImplementedError

    def config(self, **kwargs):
        ''' (Re)config the attack.

        :param kwargs: The key word arguments for the attack method's ``config()`` method.
        '''
        if self.attack_name in ('bim', 'pgd', 'mim', 'cw', 'deepfool'):
            kwargs['iteration'] = self.iteration
        else:
            kwargs['max_queries'] = self.iteration
        self.attack.config(**kwargs)

    def _run_basic(self, dataset, logger):
        ''' The ``run`` method for bim, pgd, mim. '''
        # the attack is already configured in `config()`
        rs = dict()

        iterator = dataset_to_iterator(dataset.batch(self.batch_size), self._session)
        for i_batch, (_, xs, ys, ys_target) in enumerate(iterator):
            g = self.attack.batch_attack(xs, ys, ys_target)
            try:
                step = 0
                while True:
                    step += 1
                    labels, dists = next(g)
                    if step in rs:
                        rs[step][0].append(labels)
                        rs[step][1].append(dists)
                    else:
                        rs[step] = ([labels], [dists])
                    if logger:
                        begin = i_batch * len(xs)
                        logger.info('n={}..{}: iteration={}'.format(begin, begin + len(xs) - 1, step))
            except StopIteration:
                pass

        for key in rs.keys():
            rs[key] = (np.concatenate(rs[key][0]), np.concatenate(rs[key][1]))

        return rs

    def _run_cw(self, dataset, logger):
        ''' The ``run`` method for cw. '''
        # the attack is already configured in `config()`
        iterations = [int(self.iteration * i / self.cw_n_points) for i in range(1, self.cw_n_points + 1)]
        rs = {step: ([], []) for step in iterations}

        iterator = dataset_to_iterator(dataset.batch(self.batch_size), self._session)
        for i_batch, (_, xs, ys, ys_target) in enumerate(iterator):
            for iteration in iterations:
                self.attack.config(iteration=iteration)
                xs_adv = self.attack.batch_attack(xs, ys, ys_target)
                labels, dists = self._session.run(self._cw_data, feed_dict={self._xs_ph: xs, self._xs_adv_ph: xs_adv})
                rs[iteration][0].append(labels)
                rs[iteration][1].append(dists)
                if logger:
                    begin = i_batch * len(xs)
                    logger.info('n={}..{}: iteration={}'.format(begin, begin + len(xs) - 1, iteration))

        for key in rs.keys():
            rs[key] = (np.concatenate(rs[key][0]), np.concatenate(rs[key][1]))

        return rs

    def _run_deepfool(self, dataset, logger):
        ''' The ``run`` method for deepfool. '''
        # the attack is already configured in `config()`
        rs = {step: ([], []) for step in range(1, self.attack.iteration + 1)}

        iterator = dataset_to_iterator(dataset.batch(self.batch_size), self._session)
        for i_batch, (_, xs, ys, ys_target) in enumerate(iterator):
            g = self.attack.batch_attack(xs, ys, ys_target)
            try:
                step = 0
                while True:
                    step += 1
                    labels, dists = next(g)
                    rs[step][0].append(labels)
                    rs[step][1].append(dists)
                    if logger:
                        begin = i_batch * len(xs)
                        logger.info('n={}..{}: iteration={}'.format(begin, begin + len(xs) - 1, step))
            except StopIteration:
                # DeepFool would early stop. Padding the remaining steps with the last step's data.
                labels, dists = rs[step - 1][0][-1], rs[step - 1][1][-1]
                for remain_step in range(step, self.attack.iteration + 1):
                    rs[remain_step][0].append(labels)
                    rs[remain_step][1].append(dists)

        for key in rs.keys():
            rs[key] = (np.concatenate(rs[key][0]), np.concatenate(rs[key][1]))

        return rs

    def _run_score_based(self, dataset, logger):
        ''' The ``run`` method for nes, spsa, nattack. '''
        # the attack is already configured in `config()`
        iterator = dataset_to_iterator(dataset, self._session)

        ts = []
        for i, (_, x, y, y_target) in enumerate(iterator):
            x_adv = self.attack.attack(x, y, y_target)
            labels, dists = self._session.run(self._score_based_data, feed_dict={self._x_ph: x, self._x_adv_ph: x_adv})
            label, dist, queries = labels[0], dists[0], self.attack.details['queries']
            ts.append((label, dist, queries))
            if logger:
                logger.info('n={}, {}'.format(i, self.attack.details))

        labels = np.array([x[0] for x in ts])
        dists = np.array([x[1] for x in ts])
        queries = np.array([x[2] for x in ts])

        return labels, dists, queries

    def _run_decision_based(self, dataset, logger):
        ''' The ``run`` method for boundary, evolutionary. '''
        # the attack is already configured in `config()`
        iterator = dataset_to_iterator(dataset.batch(self.batch_size), self._session)

        def pred_fn(xs):
            return self._session.run(self._xs_label, feed_dict={self._xs_ph: xs})

        cache = dict()

        rs = dict()
        for i_batch, (_, xs, ys, ys_target) in enumerate(iterator):
            starting_points = gen_starting_points(
                self.model, ys, ys_target, self.goal, self.dataset_name, self._session, pred_fn, cache)
            self.attack.config(starting_points=starting_points)

            g = self.attack.batch_attack(xs, ys, ys_target)
            try:
                step = 0
                while True:
                    step += 1
                    labels, dists = next(g)
                    if step in rs:
                        rs[step][0].append(labels)
                        rs[step][1].append(dists)
                    else:
                        rs[step] = ([labels], [dists])
                    if logger:
                        begin = i_batch * len(xs)
                        logger.info('n={}..{}: iteration={}'.format(begin, begin + len(xs) - 1, step))
            except StopIteration:
                pass

        for key in rs.keys():
            rs[key] = (np.concatenate(rs[key][0]), np.concatenate(rs[key][1]))

        return rs

    def run(self, dataset, logger=None):
        ''' Run the attack on the dataset.

        :param dataset: A ``tf.data.Dataset`` instance, whose first element is the unique identifier for the data point,
            second element is the image, third element is the ground truth label. If the goal is 'tm' or 't', a forth
            element should be provided as the target label for the attack.
        :param logger: A standard logger.
        :return:
            - For nes, spsa, nattack: Three numpy array. The first one is the labels of the adversarial examples. The
              second one is the distance between the adversarial examples and the original examples. The third one is
              queries used for attacking each examples.
            - Others: A dictionary, whose keys are iteration number, values are a tuple of two numpy array. The first
              element of the tuple is the prediction labels for the adversarial examples. The second element of the
              tuple is the distance between the adversarial examples and the original examples.
        '''
        return self._run(dataset, logger)
