''' The Loss interface. '''

from abc import ABCMeta, abstractmethod


class Loss(metaclass=ABCMeta):
    ''' An abstract interface for loss function, which is supported by some attack methods. '''

    @abstractmethod
    def __call__(self, xs, ys):
        ''' Calculate the loss with xs (input images) and ys (labels).

        :param xs: the input examples.
        :param ys: the input examples' labels.
        :return: a tensor of float number with same shape as ``ys``.
        '''
