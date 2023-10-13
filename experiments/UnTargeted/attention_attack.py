import torch
import os
from data import get_NIPS17_loader
from experiments.bard.FeatureExtractors import BlipFeatureExtractor, ClipFeatureExtractor, \
    EnsembleFeatureLoss
from utils import save_image, show_image
from attacks import SpectrumSimulationAttack, SSA_CommonWeakness, MI_CommonWeakness, MI_FGSM
from tqdm import tqdm
from utils.plot.CommonFigures import tensor_heatmap
from optimizer.losses import UnNormalizedSimpleCKA
from torch.nn import functional as F

loader = get_NIPS17_loader(batch_size=1)

blip = BlipFeatureExtractor().eval().cuda().requires_grad_(False)
clip = ClipFeatureExtractor().eval().cuda().requires_grad_(False)
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
                                  feature_loss=lambda x, y: torch.sum(x.squeeze()[:, :14, :14]))

attacker = SSA_CommonWeakness(models, epsilon=16 / 255, step_size=1 / 255, total_step=50,
                              criterion=ssa_cw_loss,
                              random_start=True)

dir = './bardadvs/'
for i, (x, y) in tqdm(enumerate(loader)):
    x = x.cuda()
    with torch.no_grad():
        before = blip(x).squeeze()
        # num_layers, head, H, D = before.shape
        head, H, D = before.shape
        before = before.reshape(-1, H, D).mean(0)
        tensor_heatmap(before, os.path.join(dir, f'{i}_before_heatmap.png'))
    ssa_cw_loss.set_ground_truth(x)
    adv_x = attacker(x, torch.zeros([0]))
    save_image(adv_x, os.path.join(dir, f'{i}.png'))
    with torch.no_grad():
        after = blip(x).squeeze()
        after = after.reshape(-1, H, D).mean(0)
        tensor_heatmap(after, os.path.join(dir, f'{i}_after_heatmap.png'))
