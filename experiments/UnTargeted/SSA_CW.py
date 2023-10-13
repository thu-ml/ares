import torch
import os
from data import get_NIPS17_loader
from experiments.bard.FeatureExtractors import BlipFeatureExtractor, ClipFeatureExtractor, \
    EnsembleFeatureLoss
from utils import save_image, show_image
from attacks import SpectrumSimulationAttack, SSA_CommonWeakness
from tqdm import tqdm

loader = get_NIPS17_loader(batch_size=1)

blip = BlipFeatureExtractor().eval().cuda().requires_grad_(False)
clip = ClipFeatureExtractor().eval().cuda().requires_grad_(False)
models = [blip, clip]


def ssa_cw_count_to_index(count, num_models=len(models), ssa_N=20):
    max = ssa_N * num_models
    count = count % max
    count = count // ssa_N
    return count


ssa_cw_loss = EnsembleFeatureLoss(models, ssa_cw_count_to_index,
                                  feature_loss=torch.nn.MSELoss())

attacker = SSA_CommonWeakness(models, epsilon=16 / 255, step_size=1 / 255, total_step=80,
                              criterion=ssa_cw_loss)

dir = './bardadvs/'
for i, (x, y) in tqdm(enumerate(loader)):
    x = x.cuda()
    ssa_cw_loss.set_ground_truth(x)
    adv_x = attacker(x, None)
    save_image(adv_x, os.path.join(dir, f'{i}.png'))
