from data import get_NIPS17_loader
from surrogates import InstructBlipVisionModel
from attacks import SpectrumSimulationAttack
from utils.ImageHandling import save_multi_images

loader = get_NIPS17_loader(batch_size=1)


class GPT4AttackCriterion:
    def __init__(self):
        self.count = 0

    def __call__(self, loss, *args):
        self.count += 1
        if self.count % 200 == 0:
            print(loss)
        return -loss


blip2 = InstructBlipVisionModel(
    target_text="bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb"
    "bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb"
    "bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb",
)
attacker = SpectrumSimulationAttack(
    [blip2],
    epsilon=64 / 255,
    step_size=1 / 255,
    total_step=2000,
    criterion=GPT4AttackCriterion(),
)
id = 0
for i, (x, y) in enumerate(loader):
    x, y = x.cuda(), y.cuda()
    adv_x = attacker(x, None)
    save_multi_images(adv_x, "./instructblip_all_bomb_advs_ssa/", begin_id=id)
    id += y.shape[0]
