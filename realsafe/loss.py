import tensorflow as tf

from abc import ABCMeta, abstractmethod


class Loss(metaclass=ABCMeta):
    ''' An abstract interface for loss function, which is supported by some attack methods. '''

    @abstractmethod
    def __call__(self, xs, ys):
        '''
        :param xs: the input examples.
        :param ys: the input examples' labels.
        :return: a tensor of float number with same shape as `ys`.
        '''
        pass


class CrossEntropyLoss(Loss):
    ''' Cross entropy loss. '''

    def __init__(self, model):
        '''
        :param model: an instance of `ClassifierWithLogits`.
        '''
        self.model = model

    def __call__(self, xs, ys):
        logits, _ = self.model.logits_and_labels(xs)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=ys,
            logits=logits,
        )
        return loss


class EnsembleCrossEntropyLoss(Loss):
    ''' Ensemble multiple models' logits, and then calculate the cross entropy loss from this ensemble logits. '''

    def __init__(self, models, weights):
        '''
        :param models: a list of `ClassifierWithLogits`.
        :param weights: weights for ensemble these models.
        '''
        self.models, self.weights = models, weights

    def __call__(self, xs, ys):
        losses = []
        for model, weight in zip(self.models, self.weights):
            logits, _ = model.logits_and_labels(xs)
            losses.append(weight * tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=ys,
                logits=logits,
            ))
        return tf.reduce_sum(losses, axis=0)


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
        logits, _ = self.model.logits_and_labels(xs)
        logits_mask = tf.one_hot(ys, self.model.n_class)
        logit_this = tf.reduce_sum(logits_mask * logits, axis=-1)
        logit_that = tf.reduce_max(logits - self.c * logits_mask, axis=-1)
        return logit_that - logit_this
