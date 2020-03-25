import inspect

from realsafe import FGSM, BIM, PGD, MIM, CW, DeepFool, NES, SPSA, NAttack, Boundary, Evolutionary


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
    '''
    Load attack method by name. The initialization function for the attack would be called by finding necessary
    parameters from `init_kwargs`.
    :param attack_name: The attack method's name. All valid values are 'fgsm', 'bim', 'pgd', 'mim', 'cw', 'deepfool',
        'nes', 'spsa', 'nattack', 'boundary', 'evolutionary'.
    :param init_kwargs: Keyword arguments for initialize the attack.
    '''
    kwargs = {}
    attack_class = ATTACKS[attack_name]
    sig = inspect.signature(attack_class.__init__)
    for name in sig.parameters:
        if name != 'self' and name in init_kwargs:
            kwargs[name] = init_kwargs[name]
    return attack_class(**kwargs)
