from .PGD import PGD


class FGSM(PGD):
    def __init__(self, *args, epsilon=16 / 255, random_start=False, **kwargs):
        kwargs['total_step'] = 1
        kwargs['random_start'] = random_start
        kwargs['epsilon'] = epsilon
        kwargs['step_size'] = epsilon
        super(FGSM, self).__init__(*args, **kwargs)
