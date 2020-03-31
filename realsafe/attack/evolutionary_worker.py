#!/usr/bin/env python3
''' MPI worker script for Evolutionary attack. '''

import sys
import collections
import cv2
import numpy as np


def attack(index, x, starting_point, y, y_target,
           x_dtype, x_shape, x_min, x_max, goal,
           mu, sigma, decay_factor, c, dimension_reduction,
           logs, xs_adv_shm):

    def fn_is_adversarial(label):
        if goal == 'ut' or goal == 'tm':
            return label != y
        else:
            return label == y_target

    def fn_mean_square_distance(x1, x2):
        return np.mean((x1 - x2) ** 2) / ((x_max - x_min) ** 2)

    x_label = yield x
    if fn_is_adversarial(x_label):
        logs.append('{}: The original image is already adversarial'.format(index))
        xs_adv_shm[index] = x
        return

    xs_adv_shm[index] = starting_point
    x_adv = starting_point
    dist = fn_mean_square_distance(x, x_adv)
    stats_adversarial = collections.deque(maxlen=30)

    if dimension_reduction:
        assert len(x_shape) == 3
        pert_shape = (*dimension_reduction, x_shape[2])
    else:
        pert_shape = x_shape

    N = np.prod(pert_shape)
    K = int(N / 20)

    evolution_path = np.zeros(pert_shape, dtype=x_dtype)
    diagonal_covariance = np.ones(pert_shape, dtype=x_dtype)

    x_adv_label = yield x_adv

    logs.append('{}: step {}, {:.5e}, prediction={}, stepsizes={:.1e}/{:.1e}: {}'.format(
        index, 0, dist, x_adv_label, sigma, mu, ''
    ))

    step = 0
    while True:
        step += 1
        unnormalized_source_direction = x - x_adv
        source_norm = np.linalg.norm(unnormalized_source_direction)

        selection_probability = diagonal_covariance.reshape(-1) / np.sum(diagonal_covariance)
        selected_indices = np.random.choice(N, K, replace=False, p=selection_probability)

        perturbation = np.random.normal(0.0, 1.0, pert_shape).astype(x_dtype)
        factor = np.zeros([N], dtype=x_dtype)
        factor[selected_indices] = 1
        perturbation *= factor.reshape(pert_shape) * np.sqrt(diagonal_covariance)

        if dimension_reduction:
            perturbation_large = cv2.resize(perturbation, x_shape[:2])
        else:
            perturbation_large = perturbation

        biased = x_adv + mu * unnormalized_source_direction
        candidate = biased + sigma * source_norm * perturbation_large / np.linalg.norm(perturbation_large)
        candidate = x - (x - candidate) / np.linalg.norm(x - candidate) * np.linalg.norm(x - biased)
        candidate = np.clip(candidate, x_min, x_max)

        candidate_label = yield candidate

        is_adversarial = fn_is_adversarial(candidate_label)
        stats_adversarial.appendleft(is_adversarial)

        if is_adversarial:
            xs_adv_shm[index] = candidate
            new_x_adv = candidate
            new_dist = fn_mean_square_distance(new_x_adv, x)
            evolution_path = decay_factor * evolution_path + np.sqrt(1 - decay_factor ** 2) * perturbation
            diagonal_covariance = (1 - c) * diagonal_covariance + c * (evolution_path ** 2)
        else:
            new_x_adv = None

        message = ''
        if new_x_adv is not None:
            abs_improvement = dist - new_dist
            rel_improvement = abs_improvement / dist
            message = 'd. reduced by {:.2f}% ({:.4e})'.format(rel_improvement * 100, abs_improvement)
            x_adv, dist = new_x_adv, new_dist
            x_adv_label = candidate_label

        logs.append('{}: step {}, {:.5e}, prediction={}, stepsizes={:.1e}/{:.1e}: {}'.format(
            index, step, dist, x_adv_label, sigma, mu, message
        ))

        if len(stats_adversarial) == stats_adversarial.maxlen:
            p_step = np.mean(stats_adversarial)
            mu *= np.exp(p_step - 0.2)
            stats_adversarial.clear()


def main():
    xs_shm_file, xs_adv_shm_file = sys.argv[1], sys.argv[2]
    batch_size = int(sys.argv[3])

    from mpi4py import MPI

    pcomm = MPI.Comm.Get_parent()

    rank = pcomm.Get_rank()

    # receive shared parameters
    shared_args = pcomm.bcast(None, root=0)
    # create memmaps
    x_shape = shared_args['x_shape']
    x_dtype = shared_args['x_dtype']
    xs_shm = np.memmap(xs_shm_file, dtype=x_dtype, mode='r+', offset=0, shape=(batch_size, *x_shape))
    xs_adv_shm = np.memmap(xs_adv_shm_file, dtype=x_dtype, mode='r+', offset=0, shape=(batch_size, *x_shape))
    # receive private parameters
    tasks = pcomm.scatter(None, root=0)

    logs = []
    attackers = []
    for task in tasks:
        index = task['index']
        attackers.append((index, attack(**task, **shared_args, logs=logs, xs_adv_shm=xs_adv_shm)))

    for index, attacker in attackers:
        xs_shm[index] = next(attacker)

    while True:
        pcomm.gather(logs, root=0)
        xs_shm_labels = pcomm.scatter(None, root=0)
        if xs_shm_labels is None:  # shutdown this worker
            break
        logs.clear()
        for label, (index, attacker) in zip(xs_shm_labels, attackers):
            try:
                xs_shm[index] = attacker.send(label)
            except StopIteration:
                pass

    pcomm.gather(['worker {} exit'.format(rank)])
    pcomm.Disconnect()


if __name__ == '__main__':
    main()
