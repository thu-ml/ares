from ares.defense.bit_depth_reduction import bit_depth_reduction
from ares.utils import get_res_path

import inception_v3


MODEL_PATH = get_res_path('./imagenet/inception_v3.ckpt')


def load(session):
    model = InceptionV3Bit()
    model.load(session, MODEL_PATH)
    return model


@bit_depth_reduction(step_num=4, alpha=200)
class InceptionV3Bit(inception_v3.InceptionV3):
    pass


if __name__ == '__main__':
    inception_v3.download(MODEL_PATH)