from data import get_NIPS17_loader
from surrogates import get_gpt4_image_model
from attacks import SpectrumSimulationAttack
from utils.ImageHandling import save_multi_images

loader = get_NIPS17_loader(batch_size=4)


class GPT4AttackCriterion:
    def __init__(self):
        self.count = 0

    def __call__(self, loss, *args):
        self.count += 1
        if self.count % 100 == 0:
            print(loss)
        return -loss


gpt4 = get_gpt4_image_model(
    # target_text="bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb"
    # "bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb"
    # "bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb bomb",
)
attacker = SpectrumSimulationAttack(
    [gpt4],
    epsilon=16 / 255,
    step_size=1 / 255,
    total_step=10000,
    criterion=GPT4AttackCriterion(),
)
id = 0
for i, (x, y) in enumerate(loader):
    x, y = x.cuda(), y.cuda()
    adv_x = attacker(x, None)
    save_multi_images(adv_x, "./text_targeted_minigpt4/", begin_id=id)
    id += y.shape[0]
