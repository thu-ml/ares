import numpy as np
import tensorflow as tf

from realsafe.benchmark.utils import load_attack, gen_starting_points
from realsafe.dataset import dataset_to_iterator


class DistortionBenchmark(object):
    ''' Distortion benchmark. '''

    def __init__(self, attack_name, model, batch_size, dataset_name, goal, distance_metric, session, init_distortion,
                 confidence, search_step=None, binsearch_step=None, **kwargs):
        ''' TODO

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
        :param init_distortion: TODO
        :param kwargs: Other keyword arguments to pass to the attack method's initialization function.
        '''
        self.init_distortion = init_distortion
        self.confidence = confidence
        self.search_step = 5
        self.binsearch_step = 10
        self.distance_metric = distance_metric

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
        self.batch_size, self.goal, self.distance_metric = batch_size, goal, distance_metric
        self.attack = load_attack(attack_name, init_kwargs)

        self._session = session

        self._xs_ph = tf.placeholder(self.model.x_dtype, shape=(None, *self.model.x_shape))

        if self.attack_name == 'fgsm':
            self._logits = self.model.logits(self._xs_ph)
            self._run = self._run_basic
        else:
            raise NotImplementedError

    def config(self, **kwargs):
        ''' (Re)config the attack method.

        :param kwargs: The key word arguments for the attack method's `config()` method.
        '''
        self.attack.config(**kwargs)

    def _run_basic(self, dataset, logger):
        ''' The `run` method for 'fgsm'. '''
        # the attack is already configured in `config()`
        rs = []

        iterator = dataset_to_iterator(dataset.batch(self.batch_size), self._session)
        for i_batch, (_, xs, ys, ys_target) in enumerate(iterator):
            # create numpy index for fetching the original and target label's logit value
            ys_range = np.arange(0, self.batch_size * self.model.n_class, self.model.n_class)
            ys_flatten = ys_range.astype(self.model.y_dtype.as_numpy_dtype) + ys
            ys_target_flatten = ys_range.astype(self.model.y_dtype.as_numpy_dtype) + ys_target
            del ys_range

            lo = np.zeros(self.batch_size, dtype=np.float32)
            hi = lo + self.init_distortion

            # set xs_result to zeros initially, so that if the attack fails all the way down we could know it.
            xs_result = np.zeros_like(xs)

            # use linear search to find an adversarial magnitude since fgsm do not play well with exponential search
            # The attack would be run with magnitude of:
            # [ init_distortion * 1, init_distortion * 2, ..., init_distortion * (2**search_steps) ].
            # The 2**search_steps here intends to archive the same sematic as exponential search.
            for i in range(2**self.search_step):
                if logger:
                    begin = i_batch * len(xs)
                    logger.info('search n={}..{}: i={}'.format(begin, begin + len(xs) - 1, i))
                magnitude = self.init_distortion * (2**self.search_step - i)
                # config the attack
                self.attack.config(magnitude=magnitude)
                # run the attack
                xs_adv = self.attack.batch_attack(xs, ys, ys_target)
                logits = self._session.run(self._logits, feed_dict={self._xs_ph: xs_adv})
                # check if attack succeed considering the confidence value
                if self.goal == 'ut' or self.goal == 'tm':
                    # for ut and tm goal, if the original label's logit is not the largest one, the example is
                    # adversarial.
                    succ = logits.max(axis=1) - logits.take(ys_flatten) > self.confidence
                else:
                    # for t goal, if the target label's logit is the largest one, the example is adversarial.
                    logits_this = logits.take(ys_target_flatten)
                    logits = logits.flatten()
                    logits[ys_target_flatten] = np.nan
                    logits_that = np.nanmax(logits.reshape((self.batch_size, -1)), axis=1)
                    succ = logits_this - logits_that > self.confidence
                # update the advsearial examples
                xs_result[succ] = xs_adv[succ]
                # update the smallest adversarial magnitude
                hi[succ] = magnitude

            lo = hi - self.init_distortion

            # run binsearch to find the minimal adversarial magnitude
            for i in range(self.binsearch_step):
                if logger:
                    begin = i_batch * len(xs)
                    logger.info('binsearch n={}..{}: i={}'.format(begin, begin + len(xs) - 1, i))
                # config the attack
                mi = (lo + hi) / 2
                self.attack.config(magnitude=mi)
                # run the attack
                xs_adv = self.attack.batch_attack(xs, ys, ys_target)
                logits = self._session.run(self._logits, feed_dict={self._xs_ph: xs_adv})
                # check if attack succeed considering the confidence value
                if self.goal == 'ut' or self.goal == 'tm':
                    # for ut and tm goal, if the original label's logit is not the largest one, the example is
                    # adversarial.
                    succ = logits.max(axis=1) - logits.take(ys_flatten) > self.confidence
                else:
                    # for t goal, if the target label's logit is the largest one, the example is adversarial.
                    logits_this = logits.take(ys_target_flatten)
                    logits = logits.flatten()
                    logits[ys_target_flatten] = np.nan
                    logits_that = np.nanmax(logits.reshape((self.batch_size, -1)), axis=1)
                    succ = logits_this - logits_that > self.confidence
                # update the advsearial examples
                xs_result[succ] = xs_adv[succ]
                # update hi (if succeed) or lo (if not)
                not_succ = np.logical_not(succ)
                hi[succ] = mi[succ]
                lo[not_succ] = mi[not_succ]

            for x, x_result in zip(xs, xs_result):
                if np.all(x_result == 0):  # all attacks failed
                    rs.append(np.nan)
                else:
                    if self.distance_metric == 'l_inf':
                        rs.append(np.max(np.abs(x_result - x)))
                    else:
                        rs.append(np.sqrt(np.sum((x_result - x)**2)))

        return np.array(rs)

    def run(self, dataset, logger=None):
        ''' Run the attack on the dataset.

        :param dataset: A `tf.data.Dataset` instance, whose first element is the unique identifier for the data point,
            second element is the image, third element is the ground truth label. If the goal is 'tm' or 't', a forth
            element should be provided as the target label for the attack.
        :param logger: A standard logger.
        :return:
        '''
        return self._run(dataset, logger)
