from realsafe.utils import get_res_path
from realsafe.defense.randomization import randomization

import resnet56

MODEL_PATH = get_res_path('./cifar10/resnet56.ckpt')


def load(session):
    model = ResNet56_Rand()
    model.load(MODEL_PATH, session)
    return model


@randomization()
class ResNet56_Rand(resnet56.ResNet56):
    pass
