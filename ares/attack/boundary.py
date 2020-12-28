import os
import sys
import tempfile

import tensorflow as tf
import numpy as np

from ares.attack.base import BatchAttack
from ares.attack.utils import split_trunks


class Boundary(BatchAttack):
    ''' Boundary. A black-box decision-based method.

    - Supported distance metric: ``l_2``.
    - Supported goal: ``t``, ``tm``, ``ut``.
    - References: https://arxiv.org/abs/1712.04248.
    '''

    def __init__(self, model, batch_size, goal, session, dimension_reduction=None, iteration_callback=None):
        ''' Initialize Boundary.

        :param model: The model to attack. A ``ares.model.Classifier`` instance.
        :param batch_size: Batch size for the ``batch_attack()`` method.
        :param goal: Adversarial goals. All supported values are ``'t'``, ``'tm'``, and ``'ut'``.
        :param session: The ``tf.Session`` to run the attack in. The ``model`` should be loaded into this session.
        :param dimension_reduction: ``(height, width)``.
        :param iteration_callback: A function accept a ``xs`` ``tf.Tensor`` (the original examples) and a ``xs_adv``
            ``tf.Tensor`` (the adversarial examples for ``xs``). During ``batch_attack()``, this callback function would
            be runned after each iteration, and its return value would be yielded back to the caller. By default,
            ``iteration_callback`` is ``None``.
        '''
        self.model, self.batch_size, self.goal, self._session = model, batch_size, goal, session

        self.dimension_reduction = dimension_reduction
        if self.dimension_reduction is not None:
            # to avoid import tensorflow in other processes, we cast the dimension to basic type
            self.dimension_reduction = (int(self.dimension_reduction[0]), int(self.dimension_reduction[1]))

        self.xs_ph = tf.placeholder(self.model.x_dtype, shape=(self.batch_size, *self.model.x_shape))
        self.xs_ph_labels = self.model.labels(self.xs_ph)

        self.iteration_callback = None
        if iteration_callback is not None:
            # store the original examples in GPU
            self.xs_var = tf.Variable(tf.zeros_like(self.xs_ph))
            self.setup_xs_var = self.xs_var.assign(self.xs_ph)
            self.iteration_callback = iteration_callback(self.xs_var, self.xs_ph)

        self.logger = None

    def config(self, **kwargs):
        ''' (Re)config the attack.

        :param starting_points: Starting points which are already adversarial. A numpy array with data type of
            ``self.x_dtype``, with shape of ``(self.batch_size, *self.x_shape)``.
        :param max_queries: Max queries. An integer.
        :param max_directions: Max directions to explore on each iteration. An integer.
        :param spherical_step: A float number.
        :param source_step: A float number.
        :param step_adaptation: A float number.
        :param maxprocs: Max number of processes to run MPI tasks. An Integer.
        :param logger: A standard logger for logging verbose information during attacking.
        '''
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

        if 'maxprocs' in kwargs:
            self.maxprocs = kwargs['maxprocs']

        if 'logger' in kwargs:
            self.logger = kwargs['logger']

    def _batch_attack_generator(self, xs, ys, ys_target):
        ''' Attack a batch of examples. It is a generator which yields back ``iteration_callback()``'s return value
        after each iteration (query) if the ``iteration_callback`` is not ``None``, and returns the adversarial
        examples.
        '''
        if self.iteration_callback is not None:
            self._session.run(self.setup_xs_var, feed_dict={self.xs_ph: xs})
        # use named memmap to speed up IPC
        xs_shm_file = tempfile.NamedTemporaryFile(prefix='/dev/shm/ares_boundary_xs_')
        xs_adv_shm_file = tempfile.NamedTemporaryFile(prefix='/dev/shm/ares_boundary_xs_adv_')
        xs_shm = np.memmap(xs_shm_file.name, dtype=self.model.x_dtype.as_numpy_dtype, mode='w+',
                           shape=(self.batch_size, *self.model.x_shape))
        xs_adv_shm = np.memmap(xs_adv_shm_file.name, dtype=self.model.x_dtype.as_numpy_dtype, mode='w+',
                               shape=(self.batch_size, *self.model.x_shape))
        # use MPI Spawn to start workers
        from mpi4py import MPI
        # use a proper number of processes
        nprocs = self.batch_size if self.batch_size <= self.maxprocs else self.maxprocs
        # since we use memmap here, run everything on localhost
        info = MPI.Info.Create()
        info.Set("host", "localhost")
        # spawn workers
        worker = os.path.abspath(os.path.join(os.path.dirname(__file__), './boundary_worker.py'))
        comm = MPI.COMM_SELF.Spawn(sys.executable, maxprocs=nprocs, info=info,
                                   args=[worker, xs_shm_file.name, xs_adv_shm_file.name, str(self.batch_size)])
        # prepare shared arguments
        shared_args = {
            'x_dtype': self.model.x_dtype.as_numpy_dtype,  # avoid importing tensorflow in workers
            'x_shape': self.model.x_shape,
            'x_min': float(self.model.x_min),
            'x_max': float(self.model.x_max),
            'goal': self.goal,
            'spherical_step': float(self.spherical_step),
            'source_step': float(self.source_step),
            'step_adaptation': float(self.step_adaptation),
            'max_queries': self.max_queries,
            'max_directions': self.max_directions,
            'dimension_reduction': self.dimension_reduction,
        }
        # prepare tasks
        all_tasks = []
        for i in range(self.batch_size):
            all_tasks.append({
                'index': i,
                'x': xs[i],
                'starting_point': self.starting_points[i],
                'y': None if ys is None else ys[i],
                'y_target': None if ys_target is None else ys_target[i],
            })
        # split tasks into trunks for each worker
        trunks = split_trunks(all_tasks, nprocs)
        # send arguments to workers
        comm.bcast(shared_args, root=MPI.ROOT)
        comm.scatter(trunks, root=MPI.ROOT)

        # the main loop
        for q in range(self.max_queries + 1):  # the first query is used to check the original examples
            # collect log from workers
            reqs = comm.gather(None, root=MPI.ROOT)
            if self.logger:
                for logs in reqs:
                    for log in logs:
                        self.logger.info(log)
            # yield back iteration_callback return value
            if self.iteration_callback is not None and q >= 1:
                yield self._session.run(self.iteration_callback, feed_dict={self.xs_ph: xs_adv_shm})
            if q == self.max_queries:
                # send a None to all workers, so that they could exit
                comm.scatter([None for _ in range(nprocs)], root=MPI.ROOT)
                reqs = comm.gather(None, root=MPI.ROOT)
                if self.logger:
                    for logs in reqs:
                        for log in logs:
                            self.logger.info(log)
            else:  # run predictions for xs_shm
                xs_ph_labels = self._session.run(self.xs_ph_labels, feed_dict={self.xs_ph: xs_shm})
                xs_ph_labels = xs_ph_labels.tolist()  # avoid pickle overhead of numpy array
                comm.scatter(split_trunks(xs_ph_labels, nprocs), root=MPI.ROOT)  # send predictions to workers
        # disconnect from MPI Spawn
        comm.Disconnect()
        # copy the xs_adv
        xs_adv = xs_adv_shm.copy()

        # delete the temp file
        xs_shm_file.close()
        xs_adv_shm_file.close()
        del xs_shm
        del xs_adv_shm

        return xs_adv

    def batch_attack(self, xs, ys=None, ys_target=None):
        ''' Attack a batch of examples.

        :return: When the ``iteration_callback`` is ``None``, return the generated adversarial examples. When the
            ``iteration_callback`` is not ``None``, return a generator, which yields back the callback's return value
            after each iteration and returns the generated adversarial examples.
        '''
        g = self._batch_attack_generator(xs, ys, ys_target)
        if self.iteration_callback is None:
            try:
                next(g)
            except StopIteration as exp:
                return exp.value
        else:
            return g
