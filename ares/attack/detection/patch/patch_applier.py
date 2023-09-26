import torch.nn as nn
from ares.utils.registry import Registry
from .patch_transform import *

class PatchApplier(nn.Module):
    """This class transforms adversarial patches and applies them to bboxes.

    Args:
        cfg (mmengine.config.ConfigDict): Configs of adversarial patches.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_transforms = self.build_transforms(training=True)
        self.test_transforms = self.build_transforms(training=False)
        self.bbox_coordinate_mode = None


    def forward(self, img_batch: torch.Tensor, adv_patch: torch.Tensor, bboxes_list: torch.Tensor,
                labels_list: [torch.Tensor]):
        """ This function transforms and applies corresponding adversarial patches for each provided bounding box.

        Args:
            img_batch (torch.Tensor): Batch image tensor. Shape: [N, C=3, H, W].
            adv_patch: Adversarial patch tensor. Shape: [num_clasess, C=3, H, W].
            bboxes_list: List of bboxes (torch.Tensor) with shape [:, 4]. Length: N.
            labels_list: List of labels (torch.Tensor) with shape [:]. Length: N.
        Returns:
            torch.Tensor: Image tensor with patches applied to. Shape: [N,C,H,W].
        """

        max_num_bboxes_per_image = max([bboxes.shape[0] for bboxes in bboxes_list])

        if max_num_bboxes_per_image == 0:
            # no gt bboxes to apply patches
            return img_batch
        adv_patch_batch, padded_bboxes = self.pad_patches_boxes(adv_patch, bboxes_list, labels_list,
                                                                max_num_bboxes_per_image)
        target_size = img_batch.shape[-2:]  # (H, W)
        if self.bbox_coordinate_mode is None:
            max_, min_ = padded_bboxes.max(), padded_bboxes.min()
            if max_ > 1.0001 and min_ >= 0.0:
                self.bbox_coordinate_mode = 'pixel'
            elif max_ <= 1.0 and min_ >= 0.0:
                self.bbox_coordinate_mode = 'normed'
            else:
                raise ValueError(f'Not supported bbox coordinate mode. Expected bbox coorninate range [0, 1] or [0, image_size], but got max value {max_}, min value {min_}')
        if self.bbox_coordinate_mode != 'pixel':
            padded_bboxes[:, :, 0::2] *= target_size[1]
            padded_bboxes[:, :, 1::2] *= target_size[0]

        if self.training:
            adv_patch_batch = self.train_transforms(adv_patch_batch, padded_bboxes, target_size)
        else:
            adv_patch_batch = self.test_transforms(adv_patch_batch, padded_bboxes, target_size)
        adv_img_batch = self.apply_patch(img_batch, adv_patch_batch)
        return adv_img_batch

    def pad_patches_boxes(self, adv_patch, bboxes_list, labels_list, max_num_bboxes_per_image):
        selected_adv_patches = []
        padded_bboxes = []
        for i in range(len(bboxes_list)):
            patches = adv_patch[labels_list[i]]
            patches = torch.cat((patches, torch.zeros((max_num_bboxes_per_image - patches.shape[0], *patches.shape[1:]),
                                                      device=patches.device)), dim=0)
            bboxes = bboxes_list[i]
            bboxes = torch.cat(
                (bboxes, torch.zeros((max_num_bboxes_per_image - bboxes.shape[0], 4), device=bboxes.device)), dim=0)
            selected_adv_patches.append(patches)
            padded_bboxes.append(bboxes)
        adv_patch_batch = torch.stack(selected_adv_patches)
        padded_bboxes = torch.stack(padded_bboxes)
        return adv_patch_batch, padded_bboxes

    def apply_patch(self, images, adv_patches):
        advs = torch.unbind(adv_patches, 1)
        for adv in advs:
            images = torch.where((adv == 0), images, adv)
        return images

    def build_transforms(self, training=True):
        transforms = []
        transform_pipeline = self.cfg.train_transforms if training else self.cfg.test_transforms
        for transform in transform_pipeline:
            name = transform['type']
            kwargs = transform['kwargs']
            if name == 'ScalePatchesToBoxes':
                kwargs.update({'size': self.cfg.size})
            transforms.append(Registry.get_transform(name)(**kwargs))
        return Compose(transforms)
