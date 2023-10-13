import torch
import os
from data import get_NIPS17_loader
from experiments.bard.FeatureExtractors import BlipFeatureExtractor, ClipFeatureExtractor, \
    EnsembleFeatureLoss
from utils import save_image
from attacks import SpectrumSimulationAttack, SSA_CommonWeakness, MI_CommonWeakness, MI_FGSM
from tqdm import tqdm
from utils.plot.CommonFigures import tensor_heatmap
from optimizer.losses import UnNormalizedSimpleCKA
from torch.nn import functional as F
from models import FixPositionPatchModel


loader = get_NIPS17_loader(batch_size=1)

blip = FixPositionPatchModel(BlipFeatureExtractor().eval().cuda().requires_grad_(False))
clip = FixPositionPatchModel(ClipFeatureExtractor().eval().cuda().requires_grad_(False))
models = [blip]


def ssa_cw_count_to_index(count, num_models=len(models), ssa_N=20):
    max = ssa_N * num_models
    count = count % max
    count = count // ssa_N
    return count


def variance_feature_loss(x, y):
    diff = x - y
    mean = torch.mean(diff)
    var = torch.var(diff)
    return mean ** 2 + var ** 2


base_cka = UnNormalizedSimpleCKA()


def cka_loss(x, y):
    x, y = x.squeeze(), y.squeeze()  # num_layers, head, H, D
    num_layers, head, H, D = x.shape
    x, y = x.view(num_layers, -1), y.view(num_layers, -1)
    return base_cka(x, y)


ssa_cw_loss = EnsembleFeatureLoss(models, ssa_cw_count_to_index,  # feature_loss=cka_loss)
                                  feature_loss=lambda x, y: torch.mean(x.mean(0).mean(0)[:, 1:2]))

attacker = MI_FGSM(models, epsilon=1, step_size=4 / 255, total_step=1000,
                   criterion=ssa_cw_loss, random_start=True)

dir = './bardadvs/'
for i, (image, y) in tqdm(enumerate(loader)):
    image, y = image.cuda(), y.cuda()
    models[0].set_images(image)
    models[0].initialize_patch(position=((0, 0, 0), (3, 28, 28)))
    x = models[0].patch
    with torch.no_grad():
        before = blip(x).squeeze()
        num_layers, head, H, D = before.shape
        # head, H, D = before.shape
        before = before.reshape(-1, H, D).mean(0)
        tensor_heatmap(before, os.path.join(dir, f'{i}_before_heatmap.png'))
    ssa_cw_loss.set_ground_truth(x)
    adv_patch = attacker(x, torch.zeros([0]))
    adv_image = models[0].add_patch_to_image(adv_patch, image, models[0].patch_position)
    save_image(adv_image, os.path.join(dir, f'{i}.png'))
    with torch.no_grad():
        after = blip(adv_patch).squeeze()
        after = after.reshape(-1, H, D).mean(0)
        print(torch.sum((after[:, 1:2] > 0.95).to(torch.float)) / H)
        print(after[:, 1:2])
        tensor_heatmap(after, os.path.join(dir, f'{i}_after_heatmap.png'))
        diff = after - before
        tensor_heatmap(diff, os.path.join(dir, f'{i}_diff_heatmap.png'))
