class BatchAttack(object):
    def __init__(self, model, batch_size):
        """
        TODO
        :param model:
        :param batch_size:
        """
        self.model = model
        self.batch_size = batch_size

    def config(self, **kwargs):
        """
        TODO
        :param kwargs:
        :return:
        """
        raise NotImplementedError("`config()` not implemented.")

    def batch_attack(self, xs, ys, ys_target, session):
        """
        TODO
        :return:
        """
        raise NotImplementedError("`batch_attack()` not implemented.")


class Attack(object):
    def __init__(self, model):
        """
        TODO
        :param model:
        """
        self.model = model

    def config(self, **kwargs):
        """
        TODO
        :param kwargs:
        :return:
        """

    def attack(self, x, y, y_target, session):
        """
        :return:
        """
