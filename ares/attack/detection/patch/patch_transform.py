import math
import numpy as np
import torch
import torch.nn.functional as F
from ares.utils.registry import Registry
from torch.nn.modules.utils import _pair, _quadruple
from torchvision.transforms import RandomHorizontalFlip

class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): List of transforms to compose.

    Example:
        >>> Compose([
        >>>     MedianPool2d(7),
        >>>     RandomJitter(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, padded_bboxes, target_size):
        for t in self.transforms:
            if isinstance(t, ScalePatchesToBoxes):
                img = t(img, padded_bboxes, target_size)
            else:
                img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

@Registry.register_transform()
class RandomHorizontalFlip(RandomHorizontalFlip):
    """See torchvision.transforms.RandomHorizontalFlip for details."""
    pass


@Registry.register_transform()
class ScalePatchesToBoxes:
    """ This class scales the given pathes to proper sizes and shifts them to the given bounding boxes
            positions in the all-zero image tensors.

    Args:
            size (int): Size of the square patch.
            scale_rate (float): Patch scale rate compared to the target bboxes sizes.
            rotate_angle (float): Max rotate angle.
            rand_shift_rate (float): Max random shift rate.
            rand_rotate (bool): Whether to randomly rotate.
            rand_shift (bool): Whether to randomly shift.
    """
    def __init__(self, size: int, scale_rate: float = 0.2,
                 rotate_angle: float = 20, rand_shift_rate: float = 0.4,
                 rand_rotate: bool = False, rand_shift: bool = False):

        super().__init__()
        self.size = size
        self.scale_rate = scale_rate
        self.rotate_angle = rotate_angle / 180 * math.pi
        self.rand_shift_rate = rand_shift_rate
        self.rand_rotate = rand_rotate
        self.rand_shift = rand_shift

    def __call__(self, adv_patch_batch, bboxes_batch, target_size):
        """Transform patches to the target size and position.

        Args:
            adv_patch_batch (torch.Tensor): Patch image tensor. Shape: [N, n, C, H, W] where n is num_bboxes_in_each_image.
            bboxes_batch (torch.Tensor): Batched bboxes tensor. Shape: [n, C, 4].
            target_size (torch.Size): Size (H, W) of target image. Size: [2].
        Returns:
            torch.Tensor: Patch image tensor. Shape: [N, n, C, H, W].
        """
        batch_size, num_bboxes_in_each_image = adv_patch_batch.shape[:2]
        H, W = target_size
        bboxes_size = batch_size * num_bboxes_in_each_image

        assert self.size <= H and self.size <= W, f'Patch size should be smaller than input image height and width, but got patch size {self.size}, image height {H} and image width {W}!'

        pad_left_right = (W - self.size) / 2
        pad_top_bottom = (H - self.size) / 2
        adv_patch_batch = F.pad(adv_patch_batch,
                                (int(pad_left_right + 0.5), int(pad_left_right),
                                 int(pad_top_bottom + 0.5), int(pad_top_bottom)),
                                'constant', value=0)  # (LRTB)

        # -------------Shift & Random relocate--------------
        # bbox format is [x1, y1, x2, y2, ...]
        bw = bboxes_batch[:, :, 2] - bboxes_batch[:, :, 0]
        bh = bboxes_batch[:, :, 3] - bboxes_batch[:, :, 1]
        target_cx = (bboxes_batch[:, :, 0] + bboxes_batch[:, :, 2]).view(bboxes_size) / 2
        target_cy = (bboxes_batch[:, :, 1] + bboxes_batch[:, :, 3]).view(bboxes_size) / 2

        if self.rand_shift:
            target_cx = self.random_shift(target_cx, bw / 2)
            target_cy = self.random_shift(target_cy, bh / 2)

        target_cx, target_cy = target_cx / W, target_cy / H
        tx = (0.5 - target_cx) * 2
        ty = (0.5 - target_cy) * 2

        # -----------------------Scale--------------------------
        # follow AdvPatch ('https://gitlab.com/EAVISE/adversarial-yolo'). You may use a different method to calculate the target size.
        # TODO: calculating scale using a new way.
        target_size = self.scale_rate * torch.sqrt((bw ** 2) + (bh ** 2)).view(bboxes_size)
        scale = target_size / self.size

        # ----------------Random Rotate-------------------------
        angle = torch.FloatTensor(bboxes_size).fill_(0).to(adv_patch_batch.device)
        if self.rand_rotate:
            angle = angle.uniform_(-self.rotate_angle, self.rotate_angle)
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # ----------Ready for the affine matrix-------------
        theta = torch.FloatTensor(bboxes_size, 2, 3).fill_(0).to(adv_patch_batch.device)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        adv_patch_batch = adv_patch_batch.view(bboxes_size, 3, adv_patch_batch.shape[-2], adv_patch_batch.shape[-1])
        grid = F.affine_grid(theta, torch.Size([bboxes_size, 3, H, W]))
        adv_patch_batch_t = F.grid_sample(adv_patch_batch, grid)

        return adv_patch_batch_t.view(batch_size, num_bboxes_in_each_image, 3, H, W)

    def random_shift(self, x, limited_range):
        shift = limited_range * torch.cuda.FloatTensor(x.size()).uniform_(-self.rand_shift_rate, self.rand_shift_rate)
        return x + shift

@Registry.register_transform()
class MedianPool2d:
    """Median pool.

    Args:
         kernel_size (int or 2-tuple): Size of pooling kernel.
         stride (int or 2-tuple): Pool stride.
         padding (int or 4-tuple (l, r, t, b)): Pool padding. It is the same as torch.nn.functional.pad.
         same (bool): Override padding and enforce same padding.
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)
        self.same = same

    def _padding(self, x: torch.Tensor):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def __call__(self, x: torch.Tensor):
        shape = x.shape
        x = x.view(-1, *shape[2:])
        assert x.dim() == 4
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        x = x.view(shape[0], shape[1], *shape[2:])
        return x

@Registry.register_transform()
class RandomJitter:
    """
    This RandomJitter class applies jitter of contrast, brightness and noise to the given tensor.

    Args:
        min_contrast (float): Min contrast.
        max_contrast (float): Max contrast.
        min_brightness (float): Min brightness.
        max_brightness (float): Max brightness.
        noise_factor (float): Noise factor.
    """

    def __init__(self, min_contrast: float = 0.8, max_contrast: float = 1.2,
                 min_brightness: float = -0.1, max_brightness: float = 0.1,
                 noise_factor: float = 0.10):
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.noise_factor = noise_factor

    def __call__(self, x: torch.Tensor):
        tensor_shape = x.shape[:2] + (1,) * (x.dim() - 2)
        contrast = torch.cuda.FloatTensor(tensor_shape).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.expand_as(x)

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(tensor_shape).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.expand_as(x)

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(x.size()).uniform_(-1, 1) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        x = contrast * x + brightness + noise
        return torch.clamp(x, 0, 1)

@Registry.register_transform()
class CutOut:
    """Cutout areas of image tensor.

    Args:
        cutout_ratio (float): Cutout area ratio of the patch.
        cutout_fill (float): Value(>0) to fill the cutout area.
        rand_shift (float): Cutout area to shift.
        level (str): Which level to randomly cut out. Supported levels: 'instance', 'batch' and 'image'.
        p_erase (float): Probability to carry out Cutout.
        verbose (bool): Whether to print information of parameters.
    """
    def __int__(self, cutout_ratio: float = 0.4, cutout_fill: float = 0.5,
                rand_shift: float = -0.05, level: str = 'instance',
                p_erase: float = 0.9, verbose: bool = False):

        self.cutout_ratio = cutout_ratio
        self.cutout_fill = cutout_fill
        self.rand_shift = rand_shift
        self.level = level
        self.p_erase = p_erase
        self.verbose = verbose

    def __call__(self, x: torch.Tensor):
        if self.verbose:
            print('Cutout level: ', self.level, '; cutout ratio: ', self.cutout_ratio, '; random shift: ',
                  self.rand_shift)

        gate = torch.tensor([0]).bernoulli_(self.p_erase)
        if gate.item() == 0: return x
        assert self.cutout_fill > 0, 'Error! The cutout area can\'t be filled with 0'
        s = x.shape
        batch_size = s[0]
        lab_len = s[1]
        bboxes_shape = s[:2] + (1,) * (x.dim() - 2)
        bboxes_size = np.prod([batch_size, lab_len])

        if self.level == "instance":
            target_size = bboxes_size
        elif self.level == "image":
            target_size = batch_size
        elif self.level == 'batch':
            target_size = 1

        bg = torch.cuda.FloatTensor(bboxes_shape).fill_(self.cutout_fill).expand_as(x)

        angle = torch.cuda.FloatTensor(target_size).fill_(0)
        if self.level != 'instance':
            angle = angle.unsqueeze(-1).expand(s[0], s[1]).reshape(-1)
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        target_cx = torch.cuda.FloatTensor(target_size).uniform_(self.rand_shift, 1 - self.rand_shift)
        target_cy = torch.cuda.FloatTensor(target_size).uniform_(self.rand_shift, 1 - self.rand_shift)
        if self.level != 'instance':
            target_cx = target_cx.unsqueeze(-1).expand(s[0], s[1]).reshape(-1)
            target_cy = target_cy.unsqueeze(-1).expand(s[0], s[1]).reshape(-1)
        tx = (0.5 - target_cx) * 2
        ty = (0.5 - target_cy) * 2

        # TODO: This assumes the patch is in a square-shape
        scale = self.cutout_ratio
        theta = torch.cuda.FloatTensor(bboxes_size, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        bg = bg.view(bboxes_size, s[2], s[3], s[4])
        x = x.view(bboxes_size, s[2], s[3], s[4])
        grid = F.affine_grid(theta, bg.shape)
        bg = F.grid_sample(bg, grid)

        x_t = torch.where((bg == 0), x, bg)
        return x_t.view(s[0], s[1], s[2], s[3], s[4])
