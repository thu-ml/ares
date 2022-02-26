"""
Implementation of attack methods. Running this file as a program will
evaluate the model and get the validation accuracy and then
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import cifar10_input
import cifar100_input
import config
from tqdm import tqdm
import os

config = config.get_args()
_NUM_RESTARTS = config.num_restarts


class LinfPGDAttack:
    def __init__(self, model, epsilon, num_steps, step_size, loss_func):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

        if loss_func == 'xent':
            loss = model.xent
        elif loss_func == 'cw':
            label_mask = tf.one_hot(model.y_input,
                                    10,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
            wrong_logit = tf.reduce_max((1 - label_mask) * model.pre_softmax - 1e4 * label_mask, axis=1)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 0)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            loss = model.xent

        self.grad = tf.gradients(loss, model.x_input)[0]

    def perturb(self, x_nat, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
        x = np.clip(x, 0, 255)

        for i in range(self.num_steps):
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                                  self.model.y_input: y})

            x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 255)  # ensure valid pixel range

        return x

def get_path_dir(data_dir, dataset, **_):
    path = os.path.join(data_dir, dataset)
    if os.path.islink(path):
        path = os.readlink(path)
    return path


if __name__ == '__main__':
    import sys
    import math
    from free_model import Model

    model_file = tf.train.latest_checkpoint(config.model_dir)
    if model_file is None:
        print('No model found')
        sys.exit()

    dataset = config.dataset
    data_dir = config.data_dir
    data_path = get_path_dir(data_dir, dataset)

    model = Model(mode='eval', dataset=dataset)
    attack = LinfPGDAttack(model,
                           config.epsilon,
                           config.pgd_steps,
                           config.step_size,
                           config.loss_func)
    saver = tf.train.Saver()


    if dataset == 'cifar10': 
      cifar = cifar10_input.CIFAR10Data(data_path)
    else:
      cifar = cifar100_input.CIFAR100Data(data_path)

    with tf.Session() as sess:
        # Restore the checkpoint
        saver.restore(sess, model_file)

        # Iterate over the samples batch-by-batch
        num_eval_examples = config.eval_examples
        eval_batch_size = config.eval_size
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        x_adv = []  # adv accumulator

        print('getting clean validation accuracy')
        total_corr = 0
        for ibatch in tqdm(range(num_batches)):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)

            x_batch = cifar.eval_data.xs[bstart:bend, :].astype(np.float32)
            y_batch = cifar.eval_data.ys[bstart:bend]

            dict_val = {model.x_input: x_batch, model.y_input: y_batch}
            cur_corr = sess.run(model.num_correct, feed_dict=dict_val)
            total_corr += cur_corr
        print('** validation accuracy: %.3f **\n\n' % (total_corr / float(num_eval_examples) * 100))

        print('Iterating over {} batches'.format(num_batches))

        total_corr, total_num = 0, 0
        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            curr_num = bend - bstart
            total_num += curr_num
            print('mini batch: {}/{} -- batch size: {}'.format(ibatch + 1, num_batches, curr_num))
            sys.stdout.flush()

            x_batch = cifar.eval_data.xs[bstart:bend, :].astype(np.float32)
            y_batch = cifar.eval_data.ys[bstart:bend]

            best_batch_adv = np.copy(x_batch)
            dict_adv = {model.x_input: best_batch_adv, model.y_input: y_batch}
            cur_corr, y_pred_batch, best_loss = sess.run([model.num_correct, model.predictions, model.y_xent],
                                                         feed_dict=dict_adv)
            for ri in range(_NUM_RESTARTS):
                x_batch_adv = attack.perturb(x_batch, y_batch, sess)
                dict_adv = {model.x_input: x_batch_adv, model.y_input: y_batch}
                cur_corr, y_pred_batch, this_loss = sess.run([model.num_correct, model.predictions, model.y_xent],
                                                             feed_dict=dict_adv)
                bb = best_loss >= this_loss
                bw = best_loss < this_loss
                best_batch_adv[bw, :, :, :] = x_batch_adv[bw, :, :, :]

                best_corr, y_pred_batch, best_loss = sess.run([model.num_correct, model.predictions, model.y_xent],
                                                              feed_dict={model.x_input: best_batch_adv,
                                                                         model.y_input: y_batch})
                print('restart %d: num correct: %d -- loss:%.4f' % (ri, best_corr, np.mean(best_loss)))
            total_corr += best_corr
            print('accuracy till now {:4}% \n\n'.format(float(total_corr) / total_num * 100))

            x_adv.append(best_batch_adv)

        x_adv = np.concatenate(x_adv, axis=0)
