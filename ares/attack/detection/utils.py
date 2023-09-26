import os
import sys
import functools

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import save_image


def normalize(tensor, mean, std):
    """Normalize input tensor with given mean and std.

    Args:
        tensor (torch.Tensor): Float tensor image of shape (B, C, H, W) to be denormalized.
        mean (torch.Tensor): Float tensor means of size (C, )  for each channel.
        std (torch.Tensor): Float tensor standard deviations of size (C, ) for each channel.
    """
    return (tensor - mean[None]) / std[None]


def denormalize(tensor, mean, std):
    """Denormalize input tensor with given mean and std.

    Args:
        tensor (torch.Tensor): Float tensor image of shape (B, C, H, W) to be denormalized.
        mean (torch.Tensor): Float tensor means of size (C, )  for each channel.
        std (torch.Tensor): Float tensor standard deviations of size (C, ) for each channel.
    """
    return tensor * std[None] + mean[None]


def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return dist.is_available() and dist.is_initialized()


def is_main_process(group=None) -> int:
    """Whether the current rank of the given process group is equal to 0.

    Note:
        Calling ``get_rank`` in non-distributed environment will return True

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        bool
    """

    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = dist.distributed_c10d._get_default_group()

        return dist.get_rank(group) == 0
    else:
        return True


def get_word_size(group=None):
    """Return the number of used GPUs."""
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = dist.distributed_c10d._get_default_group()
        return dist.get_world_size(group)
    else:
        return 1


def main_only(func):
    """Decorate those methods which should be executed in main process.

    Args:
        func (callable): Function to be decorated.

    Returns:
        callable: Return decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper


@main_only
def mkdirs_if_not_exists(dir):
    """Make dirs if it does not exist."""
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_patches_to_images(patches, save_dir, class_names, labels=None):
    """Save adversarial patches to images.

    Args:
        patches (torch.Tensor): Aderversarial patches with Shape [N,C=3,H,W].
        save_dir (str): Path to save adversarial patches.
        class_names (str): Names of classes corresponding to patches.
        labels (torch.Tensor): Labels of patches.
    """
    mkdirs_if_not_exists(save_dir)
    if not labels:
        labels = torch.arange(len(class_names))
    for cls_name, label in zip(class_names, labels):
        patch = patches[label]
        file_name = cls_name + '.png'
        save_image(patch, os.path.join(save_dir, file_name))


def save_images(img_tensors, data_samples, save_dir, with_bboxes=True, width=5, scale=True):
    """Save images.

    Args:
        img_tensors (torch.Tensor): Image tensor with shape [N,C,H,W] and value range [0, 1].
        data_samples (list): List of mmdet.structures.DetDataSample.
        save_dir (str): Path to save images.
        with_bboxes (bool): Whether to save images with bbox rectangles on images.
        width (int): Line width to draw rectangles.
        scale (bool): Whethe to scale images to original size.
    """
    mkdirs_if_not_exists(save_dir)
    for img, data_sample in zip(img_tensors, data_samples):
        img_shape = data_sample.img_shape  # (H, W)
        img = img[:, :img_shape[0], :img_shape[1]] * 255
        img = img.int().to(torch.uint8)
        img_name = os.path.basename(data_sample.img_path)
        if with_bboxes:
            bboxes = data_sample.pred_instances.bboxes.clone()
            scale_w, scale_h = data_sample.scale_factor
            bboxes[:, 1::2] *= scale_w
            bboxes[:, 0::2] *= scale_h
            img = draw_bounding_boxes(img, bboxes, width=width)
        if scale:
            ori_shape = data_sample.ori_shape
            img = F.interpolate(img[None], size=ori_shape, align_corners=True, mode='bilinear')[0]
        save_image(img / 255, os.path.join(save_dir, img_name))


class HiddenPrints:
    """Context manager to shield the output of print functions"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def tv_loss(images, reduction='mean'):
    """Implementation of the total variation loss (L_{tv}) proposed in the arxiv paper
    "Fooling automated surveillance cameras: adversarial patches to attack person detection".

    Args:
        images (torch.Tensor): Image tensor with shape [N, C, H, W] where N, C, H and W are the number of images, channel, height and width.
        reduction (str): Supported reduction methods are mean, sum and none.
    Returns:
        torch.Tensor
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)
    assert images.dim() == 4, 'Input tensor should be 4-dim, but got {%d}-dim'.format(images.dim())
    N, C, H, W = images.shape
    tv_column = torch.abs(images[..., 1:] - images[..., :-1] + 0.000001).view(N, -1)
    tv_column = torch.mean(tv_column, dim=1)
    tv_row = torch.abs(images[:, :, 1:] - images[:, :, :-1] + 0.000001).view(N, -1)
    tv_row = torch.mean(tv_row, dim=1)
    tv_loss = tv_column + tv_row
    if reduction == 'mean':
        return tv_loss.mean()
    elif reduction == 'sum':
        return tv_loss.sum()
    elif reduction == 'none':
        return tv_loss

def modify_test_pipeline(cfg):
    """The default pipeline for testing in mmdet is usually as follows:
    "LoadImageFromFile-->Resize-->LoadAnnotations-->PackDetInputs", which makes the gt bboxes are not resized.
    To resize bboxes also when resizing images, we move the "LoadAnnotations" before "Resize".
    """
    pipeline = cfg.test_dataloader.dataset.pipeline
    pop_idx = None
    for i, transform in enumerate(pipeline):
        if transform.type == 'LoadAnnotations':
            pop_idx = i
            break
    if pop_idx:
        # move LoadAnnotations before Resize
        t = pipeline.pop(pop_idx)
        pipeline.insert(1, t)

def modify_train_pipeline(cfg):
    """Modify some dataset settings in train dataloader to that in test dataloader."""
    modified_keys = ['data_root', 'ann_file', 'data_prefix']
    for key in modified_keys:
        if cfg.train_dataloader.dataset.get('dataset'):
            cfg.train_dataloader.dataset.dataset[key] = cfg.test_dataloader.dataset[key]
        else:
            cfg.train_dataloader.dataset[key] = cfg.test_dataloader.dataset[key]
    if cfg.train_dataloader.dataset.get('dataset'):
        cfg.train_dataloader.dataset.dataset.filter_cfg = dict(filter_empty_gt=True)
    else:
        cfg.train_dataloader.dataset.filter_cfg = dict(filter_empty_gt=True)

def build_optimizer(params, **kwargs):
    """Build optimizer."""
    # TODO: Add more optimizers.
    __factory__ = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD}
    return __factory__[kwargs['type']](params, **kwargs['kwargs'])

def all_reduce(tensor, reduction='sum'):
    """Gather all tensor results across all GPUs if ddp."""
    if not is_distributed():
        return
    op_factory = {'sum':dist.ReduceOp.SUM, 'avg':dist.ReduceOp.AVG}
    assert reduction in op_factory, f'Expected reductions are none, sum and mean, but got {reduction} instead!'
    op = op_factory[reduction.lower()]
    dist.all_reduce(tensor, op)

class EnableLossCal():
    """This context manager is to calculate loss for detectors from mmdet in eval mode as in training mode."""
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.pre_training = self.model.training
    def __enter__(self):
        self.model.training = True
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.training = self.pre_training