from abc import ABCMeta, abstractmethod
import tensorflow as tf


class Loss(metaclass=ABCMeta):
    """
    TODO
    """

    @abstractmethod
    def __call__(self, xs_ph, ys_ph):
        pass


class CrossEntropyLoss(Loss):
    def __init__(self, model):
        self.model = model

    def __call__(self, xs_ph, ys_ph):
        lgs, lbs = self.model.logits_and_labels(xs_ph)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=ys_ph,
            logits=lgs
        )
        return loss


class EnsembleCrossEntropyLoss(Loss):
    def __init__(self, models):
        self.models = models

    def __call__(self, xs_ph, ys_ph):
        pass
