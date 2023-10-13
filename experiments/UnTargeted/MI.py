import torch
import os
from data import get_NIPS17_loader
from experiments.bard.FeatureExtractors import BlipFeatureExtractor, ClipFeatureExtractor
from utils import save_image, show_image
from attacks import MI_FGSM
from torch import nn

loader = get_NIPS17_loader(batch_size=1)

blip = BlipFeatureExtractor().eval().cuda().requires_grad_(False)

attacker = MI_FGSM([blip], epsilon=16 / 255, step_size=1 / 255, total_step=100000,
                   criterion=nn.MSELoss(), random_start=True)

dir = './bardadvs/'
for i, (x, y) in enumerate(loader):
    x = x.cuda()
    y = None
    with torch.no_grad():
        y = blip(x)
    adv_x = attacker(x, y)
    save_image(adv_x, os.path.join(dir, f'{i}.png'))
