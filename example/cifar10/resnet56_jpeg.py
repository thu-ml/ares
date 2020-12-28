from realsafe.utils import get_res_path
from realsafe.defense.jpeg_compression import jpeg_compression

import resnet56

MODEL_PATH = get_res_path('./cifar10/resnet56.ckpt')


def load(session):
    model = ResNet56_JPEG()
    model.load(MODEL_PATH, session)
    return model


@jpeg_compression()
class ResNet56_JPEG(resnet56.ResNet56):
    pass


if __name__ == '__main__':
    resnet56.download(MODEL_PATH)