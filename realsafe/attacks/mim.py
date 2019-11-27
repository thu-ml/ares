import tensorflow as tf
import numpy as np

from realsafe.attacks.base import BatchAttack
from realsafe.attacks.utils import get_xs_ph, get_ys_ph, maybe_to_array, get_unit


class MIM(BatchAttack):
    """
    Momentum Iterative Method (MIM)
    A white-box iterative constraint-based method. Require a differentiable loss function.

    Supported distance metric: `l_2`, `l_inf`
    Supported goal: `t`, `tm`, `ut`
    Supported config parameters:
    - `magnitude`: max distortion, could be either a float number or a numpy float number array with shape of
        (batch_size,).
    - `alpha`: step size for each iteration, could be either a float number or a numpy float number array with shape of
        (batch_size,).
    - `iteration`: an integer, represent iteration count.

    References:
    [1] https://arxiv.org/abs/1710.06081
    """

    def __init__(self, model, batch_size, loss, goal, distance_metric, session):
        pass

    def config(self, **kwargs):
        pass

    def batch_attack(self, xs, ys=None, ys_target=None):
        pass
