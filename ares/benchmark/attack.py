import numpy as np
import tensorflow as tf

from ares.benchmark.utils import load_attack, gen_starting_points
from ares.dataset import dataset_to_iterator


class AttackBenchmark(object):
    ''' Run an attack on some model and report results. '''

    def __init__(self, attack_name, model, batch_size, dataset_name, goal, distance_metric, session, **kwargs):
        ''' Initialize AttackBenchmark.

        :param attack_name: The attack method's name. All valid values are ``'fgsm'``, ``'bim'``, ``'pgd'``, ``'mim'``,
            ``'cw'``, ``'deepfool'``, ``'nes'``, ``'spsa'``, ``'nattack'``, ``'boundary'``, ``'evolutionary'``.
        :param model: The classifier model to run the attack on.
        :param batch_size: Batch size for attack.
        :param dataset_name: The dataset's name. All valid values are ``'cifar10'`` and ``'imagenet'``.
        :param goal: The adversarial goal for the attack method. All valid values are ``'t'`` for targeted attack,
            ``'tm'`` for targeted missclassification attack, and ``'ut'`` for untargeted attack.
        :param distance_metric: The adversarial distance metric for the attack method. All valid values are ``'l_2'``
            and ``'l_inf'``.
        :param session: The ``tf.Session`` instance for the attack to run in.
        :param kwargs: Other keyword arguments to pass to the attack method's initialization function.
        '''
        init_kwargs = dict()
        init_kwargs['model'] = model
        init_kwargs['batch_size'] = batch_size
        init_kwargs['samples_batch_size'] = batch_size  # use same batch_size for nes, spsa, nattack's samples
        init_kwargs['goal'] = goal
        init_kwargs['distance_metric'] = distance_metric
        init_kwargs['session'] = session
        for k, v in kwargs.items():
            init_kwargs[k] = v

        self.model = model
        self.attack_name, self.dataset_name = attack_name, dataset_name
        self.batch_size, self.goal, self.distance_metric, self.session = batch_size, goal, distance_metric, session
        self.attack = load_attack(attack_name, init_kwargs)

        self.xs_ph = tf.placeholder(model.x_dtype, shape=(None, *model.x_shape))
        self.xs_label = model.labels(self.xs_ph)

    def config(self, **kwargs):
        ''' (Re)config the attack.

        :param kwargs: The key word arguments for the attack method's ``config()`` method.
        '''
        self.attack.config(**kwargs)

    def run(self, dataset, logger):
        ''' Run the attack on the dataset.

        :param dataset: A ``tf.data.Dataset`` instance, whose first element is the unique identifier for the data point,
            second element is the image, third element is the ground truth label. If the goal is 'tm' or 't', a forth
            element should be provided as the target label for the attack.
        :param logger: A standard logger.
        :return: A tuple of five numpy array. The first element represents whether the model predicting correctly on
            each dataset point. The second element represents whether the model predicting correctly on the adversarial
            example for each dataset point. The third element represents whether the dataset point is non-adversarial
            according the the goal. The fourth element represents whether the attack succeed. The fifth element
            is the generated adversarial example's distance to the dataset's original example.
        '''
        acc, acc_adv, total, succ, dist = [], [], [], [], []

        def update(accs, accs_adv, totals, succs, dists):
            acc.append(accs)
            acc_adv.append(accs_adv)
            total.append(totals)
            succ.append(succs)
            dist.append(dists)
            if logger is not None:
                logger.info('acc={:3f}, adv_acc={:3f}, succ={:3f}, dist_mean={:3f}'.format(
                    np.mean(accs.astype(np.float)), np.mean(accs_adv.astype(np.float)),
                    np.sum(succs.astype(np.float)) / np.sum(totals.astype(np.float)), np.mean(dists)
                ))

        if self.attack_name in ('fgsm', 'bim', 'pgd', 'mim', 'cw', 'deepfool'):
            iterator = dataset_to_iterator(dataset.batch(self.batch_size), self.session)
            for _, xs, ys, ts in iterator:
                xs_adv = self.attack.batch_attack(xs, ys, ts)
                xs_pred = self.session.run(self.xs_label, feed_dict={self.xs_ph: xs})
                xs_adv_pred = self.session.run(self.xs_label, feed_dict={self.xs_ph: xs_adv})
                update(*self._batch_info(xs, xs_adv, ys, ts, xs_pred, xs_adv_pred))

        elif self.attack_name in ('boundary', 'evolutionary'):
            cache = dict()
            iterator = dataset_to_iterator(dataset.batch(self.batch_size), self.session)

            def pred_fn(xs):
                return self.session.run(self.xs_label, feed_dict={self.xs_ph: xs})

            for _, xs, ys, ts in iterator:
                starting_points = gen_starting_points(
                    self.model, ys, ts, self.goal, self.dataset_name, self.session, pred_fn, cache)
                self.config(starting_points=starting_points)
                xs_adv = self.attack.batch_attack(xs, ys, ts)
                xs_pred = self.session.run(self.xs_label, feed_dict={self.xs_ph: xs})
                xs_adv_pred = self.session.run(self.xs_label, feed_dict={self.xs_ph: xs_adv})
                update(*self._batch_info(xs, xs_adv, ys, ts, xs_pred, xs_adv_pred))

        elif self.attack_name in ('nes', 'spsa', 'nattack'):
            iterator = dataset_to_iterator(dataset, self.session)
            for _, x, y, t in iterator:
                x_adv = self.attack.attack(x, y, t)
                x_pred = self.session.run(self.xs_label, feed_dict={self.xs_ph: [x]})[0]
                x_adv_pred = self.session.run(self.xs_label, feed_dict={self.xs_ph: [x_adv]})[0]
                xs, xs_adv, ys, ts = np.array([x]), np.array([x_adv]), np.array([y]), np.array([t])
                xs_pred, xs_adv_pred = np.array([x_pred]), np.array([x_adv_pred])
                update(*self._batch_info(xs, xs_adv, ys, ts, xs_pred, xs_adv_pred))

        return tuple(map(np.concatenate, (acc, acc_adv, total, succ, dist)))

    def _distance(self, xs):
        ''' Calculate distance according to the distance metric. '''
        xs = xs.reshape((xs.shape[0], -1))
        if self.distance_metric == 'l_2':
            return np.linalg.norm(xs, axis=1)
        else:
            return np.max(np.abs(xs), axis=1)

    def _batch_info(self, xs, xs_adv, ys, ts, xs_pred, xs_adv_pred):
        ''' Get benchmark information for a batch of examples. '''
        dists = self._distance(xs - xs_adv)
        accs = np.equal(xs_pred, ys)
        accs_adv = np.equal(xs_adv_pred, ys)
        if self.goal in ('tm', 'ut'):
            totals = np.equal(xs_pred, ys)
            succs = np.logical_and(totals, np.not_equal(xs_adv_pred, ys))
        else:
            totals = np.not_equal(xs_pred, ts)
            succs = np.logical_and(totals, np.equal(xs_adv_pred, ts))
        return accs, accs_adv, totals, succs, dists
