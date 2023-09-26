import os
import torch
import numpy as np
import torch.nn as nn
from ares.utils.registry import Registry
from mmengine.structures import InstanceData

from .patch.patch_applier import PatchApplier
from .utils import EnableLossCal
from .utils import normalize, denormalize, main_only
from .utils import tv_loss, mkdirs_if_not_exists, save_patches_to_images


class UniversalAttacker(nn.Module):
    '''Class supports both global perturbation attack and patch attack.

    Args:

        cfg (mmengine.config.ConfigDict): Configs for adversarial attack.
        detector (torch.nn.Module): Detector to be attacked.
        logger (logging.Logger): Logger to record logs.
        device (torch.device): torch.device. Default: torch.device(0).
    '''
    def __init__(self, cfg, detector, logger, device=torch.device(0)):
        super().__init__()

        self.cfg = cfg
        self.logger = logger
        self.detector = detector
        self.load_detector_weight()
        self.data_preprocessor = detector.data_preprocessor
        self.device = device
        self.detector_image_max_val = None
        if self.cfg.attack_mode == 'patch':
            self.init_for_patch_attack()
        elif self.cfg.attack_mode == 'global':
            self.init_for_global_attack()
        else:
            raise ValueError('Supported attack modes are patch or global, but got %s instead.' % self.cfg.attack_mode)

    def forward(self, batch_data, return_adv_images_only=False):
        '''

        Args:
            batch_data (dict): Input batch data. Example: {'inputs': torch.Tensor with shape [N,C,H,W],
                'data_samples':list of mmdet.structures.det_data_sample.DetDataSample with length N.
                }
            return_adv_images_only (bool): Whether to return adv images only without bboxes prediction.
                Default: False
        Returns:
            dict. It may contain keys losses, adv_images.
        '''
        batch_data = self.data_preprocessor(batch_data)
        if self.cfg.attack_mode == 'patch':
            return self.patch_forward(batch_data, return_adv_images_only)
        elif self.cfg.attack_mode == 'global':
            return self.global_forward(batch_data, return_adv_images_only)
        else:
            raise ValueError('Supported attack modes are patch or global, but got %s instead.' % self.cfg.attack_mode)

    def global_forward(self, batch_data, return_adv_images_only=False):
        '''For global perturbation attack.'''
        # denormalize images to range [0, 1]
        images = batch_data['inputs']
        images = denormalize(images, self.data_preprocessor.mean, self.data_preprocessor.std)
        # set image value range. We suppose the range is 0-1 or 0-255.
        if self.detector_image_max_val is None:
            max_val, min_val = images[0][0].max(), images[0][0].min()
            if max_val > 1.5 and min_val >= 0:
                self.detector_image_max_val = 255.0
            elif max_val <= 1 and min_val >= 0:
                self.detector_image_max_val = 1.0
            else:
                raise ValueError(f"Expected image pixel value range before normalization is [0, 1] or [0, 255], but got min value {min_val}, max value {max_val}!")

        images = images / self.detector_image_max_val
        batch_data['inputs'] = images
        if self.cfg.object_vanish_only:
            self.set_gt_ann_empty(batch_data['data_samples'])
        with torch.enable_grad():
            with EnableLossCal(self.detector):
                adv_images = self.attack_method.attack_detection_forward(batch_data, self.cfg.loss_fn.get('excluded_losses', []),
                                                                         self.detector_image_max_val,
                                                                         self.cfg.object_vanish_only)
        if return_adv_images_only:
            return {'adv_images': adv_images}
        # normalize adv images for detector input
        normed_adv_images = normalize(adv_images * self.detector_image_max_val, self.data_preprocessor.mean,
                                      self.data_preprocessor.std)
        preds = self.bbox_predict({'inputs':normed_adv_images, 'data_samples':batch_data['data_samples']},
                                  need_preprocess=False)
        returned_dict = {'preds': preds, 'adv_images': adv_images}
        return returned_dict

    def patch_forward(self, batch_data, return_adv_images_only=False):
        '''For patch attack'''
        # denormalize images to range [0, 1]
        images = batch_data['inputs']
        images = denormalize(images, self.data_preprocessor.mean, self.data_preprocessor.std)
        # set image value range. We suppose the range is 0-1 or 0-255.
        if self.detector_image_max_val is None:
            max_val, min_val = images[0][0].max(), images[0][0].min()
            if max_val > 1.5 and min_val >= 0:
                self.detector_image_max_val = 255.0
            elif max_val <= 1 and min_val >= 0:
                self.detector_image_max_val = 1.0
            else:
                raise ValueError(
                    f"Expected image pixel value range before normalization is [0, 1] or [0, 255], but got min value {min_val}, max value {max_val}!")
        images = images / self.detector_image_max_val
        bboxes_list, labels_list = [], []
        for i, data in enumerate(batch_data['data_samples']):
            bboxes = data.gt_instances.bboxes.clone()
            labels = data.gt_instances.labels

            if self.attacked_labels is None:
                bboxes_list.append(bboxes)
                labels_list.append(labels)
            else:
                mask = (labels[:, None] == self.attacked_labels).any(dim=1)
                bboxes_list.append(bboxes[mask])
                labels_list.append(labels[mask])

        adv_images = self.patch_applier(images, self.patch, bboxes_list, labels_list)
        if return_adv_images_only:
            return {'adv_images': adv_images}

        normed_adv_images = normalize(adv_images * self.detector_image_max_val, self.data_preprocessor.mean, self.data_preprocessor.std)
        if self.training:
            if self.cfg.object_vanish_only:
                self.set_gt_ann_empty(batch_data['data_samples'])
            detector_losses = self.detector.loss(normed_adv_images, batch_data['data_samples'])
            attacked_detector_loss = self.filter_loss(detector_losses)
            losses = {'loss_detector': attacked_detector_loss}
            if self.cfg.loss_fn.tv_loss.enable:
                selected_patch_indices = torch.cat((labels_list)).unique()
                selected_patches = self.patch[selected_patch_indices]
                loss_tv = tv_loss(selected_patches)
                loss_tv = torch.max(self.cfg.loss_fn.tv_loss.tv_scale * loss_tv, torch.tensor(self.cfg.loss_fn.tv_loss.tv_thresh).to(loss_tv.device))
                losses.update({'loss_tv': loss_tv})
            return losses
        else:
            preds = self.bbox_predict({'inputs':normed_adv_images, 'data_samples':batch_data['data_samples']},
                                      need_preprocess=False)
            returned_dict = {'preds': preds, 'adv_images': adv_images}
            return returned_dict

    def init_for_patch_attack(self):
        '''Initialize adversarial patch, patch applier and attacked labels for patch attack.'''
        self.patch = self.init_patch(init_mode=self.cfg.patch.init_mode)
        self.patch_applier = PatchApplier(self.cfg.patch)
        self.attacked_labels = self.cfg.get('attacked_labels', None)
        if self.attacked_labels:
            self.attacked_labels = torch.Tensor(self.attacked_labels).to(self.device)

    def init_for_global_attack(self):
        '''Initialize attack method for global attack.'''
        norm_type = self.cfg.attack_method.kwargs.norm
        if norm_type == 'l2':
            self.cfg.attack_method.kwargs.norm = 2
        elif norm_type == 'inf':
            self.cfg.attack_method.kwargs.norm = np.inf
        else:
            raise ValueError('Only l2 and inf norm are supported, bu got %s instead' % norm_type)
        self.attack_method = Registry.get_attack(self.cfg.attack_method.type)(self.detector, device=self.device,
                                                                              **self.cfg.attack_method.kwargs)
    def set_gt_ann_empty(self, data_samples):
        '''Set gt bboxes and gt labels zero tensors for object_vanish_only goal.'''
        bboxes = torch.zeros((1, 4), dtype=torch.float32, device=self.device)
        labels = torch.zeros((1,), dtype=torch.long, device=self.device)
        empty_gt = InstanceData(bboxes=bboxes, labels=labels, metainfo={})
        for data_sample in data_samples:
            data_sample.gt_instances = empty_gt

    def filter_loss(self, losses):
        '''Collect losses not in self.cfg.loss_fn.excluded_losses.'''
        loss_list = []
        for key in losses.keys():
            if isinstance(losses[key], list):
                losses[key] = torch.stack(losses[key]).mean()
            kept = True
            for excluded_loss in self.cfg.loss_fn.excluded_losses:
                if excluded_loss in key:
                    kept = False
                    continue
            if kept and 'loss' in key:
                loss_list.append(losses[key].mean().unsqueeze(0))
        if self.cfg.object_vanish_only:
            loss = torch.stack(loss_list).mean()
        else:
            loss = -torch.stack(loss_list).mean()
        return loss

    def freeze_layers(self, modules):
        '''Freeze given modules via setting their requires_grad attribute False.'''
        for _, parameter in modules.named_parameters():
            parameter.requires_grad = False

    def init_patch(self, init_mode='gray'):
        '''Initialize adversarial patch with given init_mode.'''
        assert init_mode in ['gray', 'white', 'black', 'random'], \
            'Expected patch initilization modes are gray, while, ' \
            'black or ramdom, bug got %s instead' % init_mode
        height = self.cfg.patch.size
        width = self.cfg.patch.size
        try:
            num_classes = self.detector.bbox_head.num_classes
        except:
            num_classes = self.detector.roi_head.bbox_head.num_classes
        self.logger.info('Adversarial patches initialzed by %s mode' % init_mode)
        if init_mode.lower() == 'random':
            patch = torch.rand((num_classes, 3, height, width))
        elif init_mode.lower() == 'gray':
            patch = torch.full((num_classes, 3, height, width), 0.5)
        elif init_mode.lower() == 'white':
            patch = torch.full((num_classes, 3, height, width), 1.0)
        elif init_mode.lower() == 'black':
            patch = torch.full((num_classes, 3, height, width), 0)
        patch = nn.Parameter(patch, requires_grad=True)
        return patch

    def load_detector_weight(self):
        '''Load detector weight from file.'''
        self.logger.info('Load detector weight from path: %s' % self.cfg.detector.weight_file)
        state_dict = torch.load(self.cfg.detector.weight_file, map_location='cpu')['state_dict']
        result = self.detector.load_state_dict(state_dict, strict=False)
        self.logger.info(result)

    def train(self, mode: bool = True):
        '''Set self to training mode.'''
        self.training = mode
        for module in self.children():
            module.train(mode)
        # detector should be set in eval model always!
        self.detector.eval()
        self.detector.training = True  # to make detector.loss() work as in training mode
        return self

    def eval(self):
        '''Set self to eval mode.'''
        self.train(False)
        self.detector.training = False
        return self

    def load_patch(self, patch_path):
        '''Initialize patch with given patch_path'''
        self.logger.info('Load adversarial patch from path: %s' % patch_path)
        patches = torch.load(patch_path, map_location=self.device)
        self.patch = torch.nn.Parameter(patches)

    @main_only
    def save_patch(self, epoch=None, is_best=False):
        '''Save adversarial patch to file.'''
        patch_save_dir = os.path.join(self.cfg.log_dir, self.cfg.patch.save_folder)
        mkdirs_if_not_exists(patch_save_dir)
        patches = self.patch.detach()

        patch_file_path = os.path.join(patch_save_dir, 'patches@epoch-' + str(epoch) + '.pth')
        patch_images_path = os.path.join(patch_save_dir, 'patch-images@epoch-' + str(epoch))

        if self.cfg.get('attacked_classes'):
            patch_classes = self.cfg.attacked_classes
            patch_labels = self.cfg.attacked_labels
        else:
            patch_classes = self.cfg.all_classes
            patch_labels = None
        torch.save(patches.cpu(), patch_file_path)
        save_patches_to_images(patches, patch_images_path, patch_classes, patch_labels)
        if is_best:
            self.logger.info('save best patches in epoch %d' % epoch)
            patch_file_path = os.path.join(patch_save_dir, 'best-patches.pth')
            patch_images_path = os.path.join(patch_save_dir, 'best-patch-images')
            torch.save(patches.cpu(), patch_file_path)
            save_patches_to_images(patches, patch_images_path, patch_classes, patch_labels)

    def bbox_predict(self, batch_data, need_preprocess=True, return_images=False):
        """

        Args:
            batch_data (dict): A dict contains inputs and data_samples attributes. See self.forward() for details.
            need_preprocess (bool): Whether to preprocess batch_data.
            return_images (bool): Whether to return input images.

        Returns:
            list or tuple : If list, return preds which is list of mmdet.structure.DetDataSample containing pred_instances attribute.
            If tuple, return (preds, images) where images are batch_data['inputs'], torch.Tensor with shape [N,C,H,W].
        """
        if need_preprocess:
            batch_data = self.data_preprocessor(batch_data)
        preds = self.detector.predict(batch_data['inputs'], batch_data['data_samples'])
        if return_images:
            images = batch_data['inputs']
            images = denormalize(images, self.data_preprocessor.mean, self.data_preprocessor.std)
            # set image value range. We suppose the range is 0-1 or 0-255.
            if self.detector_image_max_val is None:
                max_val, min_val = images[0][0].max(), images[0][0].min()
                if max_val > 1.5 and min_val >= 0:
                    self.detector_image_max_val = 255.0
                elif max_val <= 1 and min_val >= 0:
                    self.detector_image_max_val = 1.0
                else:
                    raise ValueError(
                        f"Expected image pixel value range before normalization is [0, 1] or [0, 255], but got min value {min_val}, max value {max_val}!")
            images = images / self.detector_image_max_val
            return preds, images
        return preds