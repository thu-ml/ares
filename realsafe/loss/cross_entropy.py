import tensorflow as tf

from realsafe.loss.base import Loss


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
