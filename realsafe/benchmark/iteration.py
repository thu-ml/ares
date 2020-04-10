import numpy as np
import tensorflow as tf

from realsafe.benchmark.utils import load_attack, gen_starting_points
from realsafe.dataset import dataset_to_iterator


class IterationBenchmark(object):
    ''' Iteration benchmark. '''

    def __init__(self, attack_name, model, batch_size, dataset_name, goal, distance_metric, session, **kwargs):
        '''
        :param attack_name: The attack method's name. All valid values are 'bim', 'pgd', 'mim', 'cw', 'deepfool', 'nes',
            'spsa', 'nattack', 'boundary', 'evolutionary'.
        :param model: The classifier model to run the attack on.
        :param batch_size: Batch size for attack.
        :param dataset_name: The dataset's name. All valid values are 'cifar10' and 'imagenet'.
        :param goal: The adversarial goal for the attack method. All valid values are 't' for targeted attack, 'tm' for
            targeted missclassification attack, and 'ut' for untargeted attack.
        :param distance_metric: The adversarial distance metric for the attack method. All valid values are 'l_2' and
            'l_inf'.
        :param session: The `tf.Session` instance for the attack to run in.
        :param kwargs: Other keyword arguments to pass to the attack method's initialization function.
        '''
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

        self._session = session

        if self.attack_name in ('bim', 'pgd', 'mim'):
            self._run = self._run_basic
        else:
            raise NotImplementedError

    def config(self, **kwargs):
        '''
        (Re)config the attack method.
        :param kwargs: The key word arguments for the attack method's `config()` method.
        '''
        self.attack.config(**kwargs)

    def _run_basic(self, dataset, logger):
        ''' The `run` method for 'bim', 'pgd', 'mim'. '''
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

    def run(self, dataset, logger=None):
        '''
        Run the attack on the dataset.
        :param dataset: A `tf.data.Dataset` instance, whose first element is the unique identifier for the data point,
            second element is the image, third element is the ground truth label. If the goal is 'tm' or 't', a forth
            element should be provided as the target label for the attack.
        :param logger: A standard logger.
        :return: TODO
        '''
        return self._run(dataset, logger)
