import torch
import os
from data import get_NIPS17_loader
from experiments.bard.FeatureExtractors import (
    BlipFeatureExtractor,
    ClipFeatureExtractor,
    EnsembleFeatureLoss,
    VisionTransformerFeatureExtractor,
)
from utils import save_image, get_list_image
from attacks import SpectrumSimulationAttack, SSA_CommonWeakness
from tqdm import tqdm
from torchvision import transforms

images = get_list_image("./resources/hbk/")
resizer = transforms.Resize((224, 224))
images = [resizer(i).unsqueeze(0) for i in images]


blip = BlipFeatureExtractor().eval().cuda().requires_grad_(False)
clip = ClipFeatureExtractor().eval().cuda().requires_grad_(False)
vit = VisionTransformerFeatureExtractor().eval().cuda().requires_grad_(False)
models = [blip, clip, blip, vit]


def ssa_cw_count_to_index(count, num_models=len(models), ssa_N=20):
    max = ssa_N * num_models
    count = count % max
    count = count // ssa_N
    return count


def variance_feature_loss(x, y):
    diff = x - y
    mean = torch.mean(diff)
    var = torch.var(diff)
    return mean**2 + var**2


# base_cka = UnNormalizedSimpleCKA()
#
#
# def cka_loss(x, y):
#     x, y = x.reshape(44, 32), y.reshape(44, 32)
#     return base_cka(x, y)


ssa_cw_loss = EnsembleFeatureLoss(models, ssa_cw_count_to_index, feature_loss=torch.nn.MSELoss())


attacker = SSA_CommonWeakness(
    models,
    epsilon=16 / 255,
    step_size=1 / 255,
    total_step=500,
    criterion=ssa_cw_loss,
)

dir = "./hbk_feature_attack/"
for i, x in enumerate(tqdm(images)):
    if i >= 100:
        break
    x = x.cuda()
    ssa_cw_loss.set_ground_truth(x)
    adv_x = attacker(x, None)
    save_image(adv_x, os.path.join(dir, f"{i}.png"))
