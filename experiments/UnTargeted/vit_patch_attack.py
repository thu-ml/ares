from models.BaseNormModel import FixPositionPatchModel, BaseNormModel
from torchvision.models import vit_b_16
from attacks import MI_FGSM
from data import get_NIPS17_loader
from utils.ImageHandling import save_image

loader = get_NIPS17_loader(batch_size=1)
model = FixPositionPatchModel(BaseNormModel(vit_b_16(pretrained=True)))
model.initialize_patch()
attacker = MI_FGSM([model], epsilon=1, step_size=4 / 255, total_step=100)
for x, y in loader:
    model.set_images(x)
    adv_x = attacker(model.patch, y)
    save_image(adv_x, './vit_patch_adv.png')
    assert False
