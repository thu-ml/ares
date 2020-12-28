import inspect
import numpy as np

from ares import FGSM, BIM, PGD, MIM, CW, DeepFool, NES, SPSA, NAttack, Boundary, Evolutionary
from ares.dataset import cifar10, imagenet, dataset_to_iterator


ATTACKS = {
    'fgsm': FGSM,
    'bim': BIM,
    'pgd': PGD,
    'mim': MIM,
    'cw': CW,
    'deepfool': DeepFool,
    'nes': NES,
    'spsa': SPSA,
    'nattack': NAttack,
    'boundary': Boundary,
    'evolutionary': Evolutionary,
}


def load_attack(attack_name, init_kwargs):
    ''' Load attack method by name. The initialization function for the attack would be called by finding necessary
    parameters from ``init_kwargs``.

    :param attack_name: The attack method's name. All valid values are ``'fgsm'``, ``'bim'``, ``'pgd'``, ``'mim'``,
        ``'cw'``, ``'deepfool'``, ``'nes'``, ``'spsa'``, ``'nattack'``, ``'boundary'``, ``'evolutionary'``.
    :param init_kwargs: Keyword arguments for initialize the attack.
    '''
    kwargs = {}
    attack_class = ATTACKS[attack_name]
    sig = inspect.signature(attack_class.__init__)
    for name in sig.parameters:
        if name != 'self' and name in init_kwargs:
            kwargs[name] = init_kwargs[name]
    return attack_class(**kwargs)


def gen_starting_points(model, ys, ys_target, goal, dataset_name, session, pred_fn, cache=None):
    ''' Generate starting points which are already adversarial according to the adversarial goal.

    :param model: The model.
    :param ys: True labels.
    :param ys_target: Targetted labels.
    :param goal: Adversarial goal.
    :param dataset_name: The dataset's name. All valid values are ``'cifar10'`` and ``'imagenet'``.
    :param session: ``tf.Session`` for loading dataset.
    :param pred_fn: A function which accepts a batch of model inputs as a numpy array and returns the model's
        predictions.
    :param cache: A cache for reusing generated starting points. A dictionary. Same cache shall not be shared between
        different model and adversarial goal.
    :return: Starting points as a numpy array.
    '''
    if cache is None:
        cache = dict()

    starting_points = np.zeros((len(ys), *model.x_shape), dtype=model.x_dtype.as_numpy_dtype)

    if goal in ('ut', 'tm'):
        for index, y in enumerate(ys):
            y = int(y)
            if y not in cache:
                while True:
                    x = np.random.uniform(low=model.x_min, high=model.x_max, size=(1, *model.x_shape))
                    x = x.astype(model.x_dtype.as_numpy_dtype)
                    x_pred = pred_fn(x)[0]
                    if x_pred != y:
                        cache[y] = x[0]
                        break
            starting_points[index] = cache[y]
    else:
        for index, y in enumerate(ys_target):
            if y not in cache:
                if dataset_name == 'cifar10':
                    dataset = cifar10.load_dataset_for_classifier(model, target_label=y).batch(1)
                else:
                    dataset = imagenet.load_dataset_for_classifier(model, target_label=y).batch(1)
                for _, x, _ in dataset_to_iterator(dataset, session):
                    x_pred = pred_fn(x)[0]
                    if x_pred == y:
                        cache[y] = x[0]
                        break
            starting_points[index] = cache[y]

    return starting_points
