from realsafe.defense.jpeg_compression import jpeg_compression
from realsafe.utils import get_res_path

import inception_v3


MODEL_PATH = get_res_path('./imagenet/inception_v3.ckpt')


def load(session):
    model = InceptionV3Jpeg()
    model.load(session, MODEL_PATH)
    return model


@jpeg_compression(quality=75)
class InceptionV3Jpeg(inception_v3.InceptionV3):
    pass
