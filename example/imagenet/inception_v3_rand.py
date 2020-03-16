from realsafe.defense.randomization import randomization

import inception_v3


def load(session):
    model = InceptionV3Rand()
    model.load(session)
    return model


@randomization()
class InceptionV3Rand(inception_v3.InceptionV3):
    pass
