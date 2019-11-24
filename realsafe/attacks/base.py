from abc import ABCMeta, abstractmethod


class BatchAttack(metaclass=ABCMeta):
    """
    TODO
    """

    @abstractmethod
    def config(self, **kwargs):
        """
        TODO
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def batch_attack(self, xs, ys, ys_target):
        """
        TODO
        :param xs:
        :param ys:
        :param ys_target:
        :return:
        """
        pass


class Attack(metaclass=ABCMeta):
    """
    TODO
    """

    @abstractmethod
    def config(self, **kwargs):
        """
        TODO
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def attack(self, x, y, y_target):
        """
        TODO
        :param x:
        :param y:
        :param y_target:
        :return:
        """
        pass
