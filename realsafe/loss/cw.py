import tensorflow as tf

from realsafe.loss.base import Loss


class CWLoss(Loss):
    ''' C&W loss. '''

    def __init__(self, model, c=99999.0):
        '''
        :param model: an instance of `ClassifierWithLogits`.
        :param c: a large float number.
        '''
        self.model = model
        self.c = c

    def __call__(self, xs, ys):
        logits = self.model.logits(xs)
        logits_mask = tf.one_hot(ys, self.model.n_class)
        logit_this = tf.reduce_sum(logits_mask * logits, axis=-1)
        logit_that = tf.reduce_max(logits - self.c * logits_mask, axis=-1)
        return logit_that - logit_this
