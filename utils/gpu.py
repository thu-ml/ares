import torch
import time


def hold_gpu(memory: float = 24):
    size = 655360000 * 9 * memory / 24
    x = torch.randn(int(size), device=torch.device('cuda'))
    time.sleep(86400 * 4)
