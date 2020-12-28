#!/usr/bin/env python3
''' MPI worker script for Boundary attack. '''

import sys
import collections
import cv2
import numpy as np


def attack(index, x, starting_point, y, y_target,
           x_dtype, x_shape, x_min, x_max, goal, spherical_step, source_step, step_adaptation, max_queries,
           max_directions, dimension_reduction,
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
    dist = fn_mean_square_distance(x_adv, x)
    stats_spherical_adversarial = collections.deque(maxlen=100)
    stats_step_adversarial = collections.deque(maxlen=30)

    x_adv_label = yield x_adv

    logs.append('{}: step {}, {:.5e}, prediction={}, stepsizes={:.1e}/{:.1e}: {}'.format(
        index, 0, dist, x_adv_label, spherical_step, source_step, ''
    ))

    step, queries = 0, 0
    while True:
        step += 1

        unnormalized_source_direction = x - x_adv
        source_norm = np.linalg.norm(unnormalized_source_direction)
        source_direction = unnormalized_source_direction / source_norm

        do_spherical = (step % 10 == 0)

        for _ in range(max_directions):
            if dimension_reduction:
                assert len(x_shape) == 3
                perturbation_shape = (*dimension_reduction, x_shape[2])
                perturbation = np.random.normal(0.0, 1.0, perturbation_shape).astype(x_dtype)
                perturbation = cv2.resize(perturbation, x_shape[:2])
            else:
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
                spherical_candidate_label = yield spherical_candidate
                spherical_is_adversarial = fn_is_adversarial(spherical_candidate_label)

                queries += 1
                if queries == max_queries:
                    return
                stats_spherical_adversarial.appendleft(spherical_is_adversarial)

                if not spherical_is_adversarial:
                    continue

            candidate_label = yield candidate
            is_adversarial = fn_is_adversarial(candidate_label)

            queries += 1
            if queries == max_queries:
                return

            if do_spherical:
                stats_step_adversarial.appendleft(is_adversarial)

            if not is_adversarial:
                continue

            new_x_adv = candidate
            xs_adv_shm[index] = new_x_adv
            new_dist = fn_mean_square_distance(new_x_adv, x)
            break
        else:
            new_x_adv = None

        message = ''
        if new_x_adv is not None:
            abs_improvement = dist - new_dist
            rel_improvement = abs_improvement / dist
            message = 'd. reduced by {:.2f}% ({:.4e})'.format(rel_improvement * 100, abs_improvement)
            x_adv_label, x_adv, dist = candidate_label, new_x_adv, new_dist

        logs.append('{}: step {}, {:.5e}, prediction={}, stepsizes={:.1e}/{:.1e}: {}'.format(
            index, step, dist, x_adv_label, spherical_step, source_step, message
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
                logs.append('{}: {} {:.2f} ({:3d}), {:.2f} ({:3d})'.format(
                    index, message, p_spherical, n_spherical, p_step, n_step
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
                logs.append('{}: {} {:.2f} ({:3d}), {:.2f} ({:3d})'.format(
                    index, message, p_spherical, n_spherical, p_step, n_step
                ))

        if queries == max_queries:
            return


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
