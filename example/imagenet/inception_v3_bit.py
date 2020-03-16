from realsafe.defense.bit_depth_reduction import bit_depth_reduction

import inception_v3


def load(session):
    model = InceptionV3Bit()
    model.load(session)
    return model


@bit_depth_reduction(step_num=4, alpha=200)
class InceptionV3Bit(inception_v3.InceptionV3):
    pass
