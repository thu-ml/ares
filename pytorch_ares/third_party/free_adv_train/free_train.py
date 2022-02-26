"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import shutil
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
import sys
from free_model import Model
import cifar10_input
import cifar100_input
import pdb

import config

def get_path_dir(data_dir, dataset, **_):
    path = os.path.join(data_dir, dataset)
    if os.path.islink(path):
        path = os.readlink(path)
    return path


def train(tf_seed, np_seed, train_steps, out_steps, summary_steps, checkpoint_steps, step_size_schedule,
          weight_decay, momentum, train_batch_size, epsilon, replay_m, model_dir, dataset, **kwargs):
    tf.set_random_seed(tf_seed)
    np.random.seed(np_seed)

    model_dir = model_dir + '%s_m%d_eps%.1f_b%d' % (dataset, replay_m, epsilon, train_batch_size)  # TODO Replace with not defaults

    # Setting up the data and the model
    data_path = get_path_dir(dataset=dataset, **kwargs)
    if dataset == 'cifar10':
      raw_data = cifar10_input.CIFAR10Data(data_path)
    else:
      raw_data = cifar100_input.CIFAR100Data(data_path)
    global_step = tf.contrib.framework.get_or_create_global_step()
    model = Model(mode='train', dataset=dataset, train_batch_size=train_batch_size)

    # Setting up the optimizer
    boundaries = [int(sss[0]) for sss in step_size_schedule][1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)

    # Optimizing computation
    total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
    grads = optimizer.compute_gradients(total_loss)

    # Compute new image
    pert_grad = [g for g, v in grads if 'perturbation' in v.name]
    sign_pert_grad = tf.sign(pert_grad[0])
    new_pert = model.pert + epsilon * sign_pert_grad
    clip_new_pert = tf.clip_by_value(new_pert, -epsilon, epsilon)
    assigned = tf.assign(model.pert, clip_new_pert)

    # Train
    no_pert_grad = [(tf.zeros_like(v), v) if 'perturbation' in v.name else (g, v) for g, v in grads]
    with tf.control_dependencies([assigned]):
        min_step = optimizer.apply_gradients(no_pert_grad, global_step=global_step)
    tf.initialize_variables([model.pert])  # TODO: Removed from TF

    # Setting up the Tensorboard and checkpoint outputs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    saver = tf.train.Saver(max_to_keep=1)
    tf.summary.scalar('accuracy', model.accuracy)
    tf.summary.scalar('xent', model.xent / train_batch_size)
    tf.summary.scalar('total loss', total_loss / train_batch_size)
    merged_summaries = tf.summary.merge_all()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print('\n\n********** free training for epsilon=%.1f using m_replay=%d **********\n\n' % (epsilon, replay_m))
        print('important params >>> \n model dir: %s \n dataset: %s \n training batch size: %d \n' % (model_dir, dataset, train_batch_size))
        if dataset == 'cifar100':
          print('the ride for CIFAR100 is bumpy -- fasten your seatbelts! \n \
          you will probably see the training and validation accuracy fluctuating a lot early in trainnig \n \
                this is natural especially for large replay_m values because we see that mini-batch so many times.')
        # initialize data augmentation
        if dataset == 'cifar10':
          data = cifar10_input.AugmentedCIFAR10Data(raw_data, sess, model)
        else:
          data = cifar100_input.AugmentedCIFAR100Data(raw_data, sess, model)

        # Initialize the summary writer, global variables, and our time counter.
        summary_writer = tf.summary.FileWriter(model_dir + '/train', sess.graph)
        eval_summary_writer = tf.summary.FileWriter(model_dir + '/eval')
        sess.run(tf.global_variables_initializer())

        # Main training loop
        for ii in range(train_steps):
            if ii % replay_m == 0:
                x_batch, y_batch = data.train_data.get_next_batch(train_batch_size, multiple_passes=True)
                nat_dict = {model.x_input: x_batch, model.y_input: y_batch}

            x_eval_batch, y_eval_batch = data.eval_data.get_next_batch(train_batch_size, multiple_passes=True)
            eval_dict = {model.x_input: x_eval_batch, model.y_input: y_eval_batch}

            # Output to stdout
            if ii % summary_steps == 0:
                train_acc, summary = sess.run([model.accuracy, merged_summaries], feed_dict=nat_dict)
                summary_writer.add_summary(summary, global_step.eval(sess))
                val_acc, summary = sess.run([model.accuracy, merged_summaries], feed_dict=eval_dict)
                eval_summary_writer.add_summary(summary, global_step.eval(sess))
                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    training nat accuracy {:.4}% -- validation nat accuracy {:.4}%'.format(train_acc * 100,
                                                                                                  val_acc * 100))
                sys.stdout.flush()
            # Tensorboard summaries
            elif ii % out_steps == 0:
                nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    training nat accuracy {:.4}%'.format(nat_acc * 100))

            # Write a checkpoint
            if (ii+1) % checkpoint_steps == 0:
                saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)

            # Actual training step
            sess.run(min_step, feed_dict=nat_dict)


if __name__ == '__main__':
    args = config.get_args()
    train(**vars(args))
