''' The Attack interface. '''

from abc import ABCMeta, abstractmethod


class BatchAttack(metaclass=ABCMeta):
    ''' An abstract interface for attack methods which support attacking a batch of input at the same time.

    All the graph construction should be done in the ``__init__()`` method. The ``config()`` method shall not create new
    tensorflow graph node, since it might be invoked lots of times during running benchmarks. If creating new graph node
    inside the ``config()`` method, large memory leakage might occur after calling this method tens of thousands of
    times.
    '''

    @abstractmethod
    def config(self, **kwargs):
        ''' (Re)config the attack.

        :param kwargs: Change the configuration of the attack method. This methods could be called multiple time, so
            that no new tensorflow graph nodes shall be created to avoid memory leak. Should support partial
            configuration, so that for each configuration option, only the newest values are kept.
        '''

    @abstractmethod
    def batch_attack(self, xs, ys=None, ys_target=None):
        ''' Generate adversarial examples from a batch of examples.

        :param xs: The original examples.
        :param ys: The original examples' ground truth labels, could be ``None``.
        :param ys_target: The targeted labels, could be ``None``.
        :return: Adversarial examples. Other detail information could be returned by storing in its ``details``
            property.
        '''


class Attack(metaclass=ABCMeta):
    ''' An abstract interface for attack methods which support only attacking one input at the same time.

    All the graph construction should be done in the ``__init__()`` method. The ``config()`` method shall not create new
    tensorflow graph node, since it might be invoked lots of times during running benchmarks. If creating new graph node
    inside the ``config()`` method, large memory leakage might occur after calling this method tens of thousands of
    times.
    '''

    @abstractmethod
    def config(self, **kwargs):
        ''' (Re)config the attack.

        :param kwargs: Change the configuration of the attack method. This methods could be called multiple time, so
            that no new tensorflow graph nodes shall be created to avoid memory leak. Should support partial
            configuration, so that for each configuration option, only the newest values are kept.
        '''

    @abstractmethod
    def attack(self, x, y=None, y_target=None):
        ''' Generate adversarial example from one example.

        :param x: The original example.
        :param y: The original example's ground truth label, could be ``None``.
        :param y_target: The targeted label, could be ``None``.
        :return: Adversarial example. Other detail information could be returned by storing in its ``details`` property.
        '''
