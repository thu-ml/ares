import tensorflow as tf
from abc import ABCMeta, abstractmethod


class Loss(metaclass=ABCMeta):
    """
    TODO
    """

    @abstractmethod
    def __call__(self, xs_ph, ys_ph):
        pass


class CrossEntropyLoss(Loss):
    """
    TODO
    """

    def __init__(self, model):
        """
        TODO
        :param model:
        """
        self.model = model

    def __call__(self, xs_ph, ys_ph):
        lgs, _ = self.model.logits_and_labels(xs_ph)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=ys_ph,
            logits=lgs
        )
        return loss


class EnsembleCrossEntropyLoss(Loss):
    """
    TODO
    """

    def __init__(self, models, weights):
        """
        TODO
        :param models:
        """
        self.models, self.weights = models, weights

    def __call__(self, xs_ph, ys_ph):
        losses = []
        for model, weight in zip(self.models, self.weights):
            lgs, _ = model.logits_and_labels(xs_ph)
            losses.append(weight * tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=ys_ph,
                logits=lgs
            ))
        return tf.reduce_sum(losses, axis=0)
