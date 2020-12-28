from ares.defense.randomization import randomization
from ares.utils import get_res_path

import inception_v3


MODEL_PATH = get_res_path('./imagenet/inception_v3.ckpt')


def load(session):
    model = InceptionV3Rand()
    model.load(session, MODEL_PATH)
    return model


@randomization()
class InceptionV3Rand(inception_v3.InceptionV3):
    pass


if __name__ == '__main__':
    inception_v3.download(MODEL_PATH)