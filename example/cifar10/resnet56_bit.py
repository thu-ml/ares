from realsafe.utils import get_res_path
from realsafe.defense.bit_depth_reduction import bit_depth_reduction

import resnet56

MODEL_PATH = get_res_path('./cifar10/resnet56.ckpt')


def load(session):
    model = ResNet56_BDR()
    model.load(MODEL_PATH, session)
    return model


@bit_depth_reduction(8)
class ResNet56_BDR(resnet56.ResNet56):
    pass
