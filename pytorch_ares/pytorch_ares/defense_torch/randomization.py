import torch
import torch.nn.functional as F


class Randomization(object):
    def __init__(self, model, device,data_name, prob=0.8, crop_lst=[0.1, 0.08, 0.06, 0.04, 0.02]):
        self.prob = prob
        self.crop_lst = crop_lst
        self.data_name = data_name
        self.model = model
        self.device = device


    def input_transform(self, xs):
        p = torch.rand(1).item()
        if p <= self.prob:
            out = self.random_resize_pad(xs)
            return out
        else:
            return xs

    def random_resize_pad(self, xs):
        rand_cur = torch.randint(low=0, high=len(self.crop_lst), size=(1,)).item()
        crop_size = 1 - self.crop_lst[rand_cur]
        pad_left = torch.randint(low=0, high=3, size=(1,)).item() / 2
        pad_top = torch.randint(low=0, high=3, size=(1,)).item() / 2

        if len(xs.shape) == 4:
            bs, c, w, h = xs.shape
        elif len(xs.shape) == 5:
            bs, fs, c, w, h = xs.shape
        w_, h_ = int(crop_size * w), int(crop_size * h)
        # out = resize(xs, size=(w_, h_))
        out = F.interpolate(xs, size=[w_, h_], mode='bicubic', align_corners=False)
        

        pad_left = int(pad_left * (w - w_))
        pad_top = int(pad_top * (h - h_))
        out = F.pad(out, [pad_left, w - pad_left - w_, pad_top, h - pad_top - h_], value=0)
        
        return out
