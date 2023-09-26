import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .custom.lr_scheduler import build_lr_scheduler
from .utils import build_optimizer
from .utils import get_word_size
from .utils import is_distributed
from .utils import mkdirs_if_not_exists
from .utils import save_images
from .utils import all_reduce

class Trainer():
    """Base trainer class.

    Args:
        cfg (mmengine.config.ConfigDict): Attack config dict.
        model (torch.nn.Module): Model to be trained or evaluated.
        train_dataloader (torch.utils.data.Dataloader): Dataloader for training.
        test_dataloader (torch.utils.data.Dataloader): Dataloader for testing.
        evaluator (class): Evaluator to evaluate detection performance.
        logger (logging.Logger): Logger to record information.
    """
    def __init__(self, cfg, model, train_dataloader, test_dataloader, evaluator, logger):
        self.cfg = cfg
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.evaluator = evaluator
        self.logger = logger
        self.epochs = self.cfg.get('epochs', 0)

        self.before_start()

    def train(self):
        """Train model."""
        self.before_train()
        for epoch in range(1, self.epochs + 1):
            self.runtime['epoch'] = epoch
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    @torch.no_grad()
    def eval(self, eval_on_clean=False):
        '''Evaluate detection performance.'''
        self.before_eval()
        if eval_on_clean:
            self.eval_clean()

        save_adv_images = self.cfg.adv_image.save

        if save_adv_images:
            adv_image_save_dir = os.path.join(self.cfg.log_dir, self.cfg.adv_image.save_folder)
            mkdirs_if_not_exists(adv_image_save_dir)
        if self.is_distributed:
            dist.barrier()
        self.logger.info('Evaluating detection performance on attacked data...')
        for i, batch_data in tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader)):
            with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                returned_dict = self.model(batch_data)
            preds = returned_dict['preds']
            if save_adv_images:
                save_images(returned_dict['adv_images'], preds, adv_image_save_dir,
                            self.cfg.adv_image.with_bboxes)
            self.evaluator.process(data_samples=preds)
        metrics = self.evaluator.evaluate(len(self.test_dataloader.dataset))
        return metrics

    def run_epoch(self):
        """Train for one epoch."""
        epoch_loss = torch.tensor(0.0, device=self.rank, requires_grad=False)
        t1 = time.time()
        for i, batch_data in enumerate(self.train_dataloader):
            with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                losses = self.model(batch_data)
            loss = sum(losses.values())
            if self.cfg.loss_fn.tv_loss.enable:
                loss_tv = losses['loss_tv'].detach()
                all_reduce(loss_tv, 'avg')

            batch_loss = loss.detach()
            all_reduce(batch_loss, 'avg')
            epoch_loss += batch_loss.item()
            self.runtime['total_loss']['value'] += batch_loss.item()
            self.runtime['total_loss']['length'] += 1
            # some batch images without gt bounding boxes may lead to no grad computed
            try:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # patch values should be in range [0, 1]
                if self.is_distributed:
                    torch.clamp(self.model.module.patch, min=0, max=1)
                else:
                    torch.clamp(self.model.patch, min=0, max=1)
            except:
                pass
            t2 = time.time()
            batch_time = t2 - t1
            t1 = t2
            if i % self.cfg.log_period == 0:
                epoch = self.runtime['epoch']
                length = len(self.train_dataloader)
                avg_loss = self.runtime['total_loss']['value'] / self.runtime['total_loss']['length']

                info = f'Epoch:{epoch:3d}/{self.epochs}  Iter:{i:4d}/{length}  loss:{batch_loss:.2f} ({avg_loss:.2f})  '
                if self.cfg.loss_fn.tv_loss.enable:
                    info += f'loss tv:{loss_tv.item():.2f}  '
                info += f'time:{batch_time:.1f}s'
                self.logger.info(info)

        self.runtime['epoch_loss'] = epoch_loss
    @torch.no_grad()
    def eval_clean(self):
        """Evaluate detection performance on clean data."""
        if self.cfg.clean_image.save:
            clean_image_save_dir = os.path.join(self.cfg.log_dir, self.cfg.clean_image.save_folder)
            mkdirs_if_not_exists(clean_image_save_dir)

        self.logger.info('Evaluating detection performance on clean data...')
        model = self.model.module if self.is_distributed else self.model
        for i, batch_data in tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader)):
            with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                preds, images = model.bbox_predict(batch_data, return_images=True)
            self.evaluator.process(data_samples=preds)
            if self.cfg.clean_image.save:
                save_images(images, preds, clean_image_save_dir,
                            self.cfg.clean_image.with_bboxes)
        self.evaluator.evaluate(len(self.test_dataloader.dataset))

    def before_eval(self):
        """Do something before evaluating."""
        self.model.eval() if not self.is_distributed else self.model.module.eval()
        self.test_dataloader.sampler.shuffle = False

    def before_epoch(self):
        """Do something before each training epoch."""
        epoch = self.runtime['epoch']
        self.model.train() if not self.is_distributed else self.model.module.train()
        if self.is_distributed:
            self.train_dataloader.batch_sampler.sampler.set_epoch(epoch)

    def after_epoch(self):
        """Do something after each training epoch."""
        epoch = self.runtime['epoch']
        all_reduce(self.runtime['epoch_loss'], reduction='sum')
        epoch_loss = self.runtime['epoch_loss'] / len(self.train_dataloader)
        self.lr_scheduler.step(loss=epoch_loss.item(), epoch=epoch)

        if epoch % self.cfg.patch.save_period == 0:
            model = self.model.module if self.is_distributed else self.model
            model.save_patch(self.runtime['epoch'], is_best=False)

        if epoch % self.cfg.eval_period == 0 and epoch != self.epochs:
            metrics = self.eval(eval_on_clean=False)
            is_best = metrics['coco/bbox_mAP'] < self.runtime['lowest_bbox_mAP']
            if is_best:
                self.runtime['lowest_bbox_mAP'] = metrics['coco/bbox_mAP']
                self.logger.info('Lowest mAP updated!')
            model = self.model.module if self.is_distributed else self.model
            model.save_patch(self.runtime['epoch'], is_best=is_best)

    def before_train(self):
        """Automatically scale learning rate, build optimizer and lr_scheduler before training."""
        self.train_dataloader.sampler.shuffle = True
        if self.is_distributed:
            params = [{'params': self.model.module.patch}]
        else:
            params = [{'params': self.model.patch}]
        self.scale_lr()
        self.optimizer = build_optimizer(params, **self.cfg.optimizer)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, **self.cfg.lr_scheduler)

    def after_train(self):
        """Do something after finishing training."""
        metrics = self.eval(eval_on_clean=False)
        if self.rank == 0:
            patch_save_dir = os.path.join(self.cfg.log_dir, self.cfg.patch.save_folder)
            mkdirs_if_not_exists(patch_save_dir)
        is_best = metrics['coco/bbox_mAP'] < self.runtime['lowest_bbox_mAP']
        if is_best:
            self.runtime['lowest_bbox_mAP'] = metrics['coco/bbox_mAP']
            self.logger.info('Lowest mAP updated!')
        model = self.model.module if self.is_distributed else self.model
        model.save_patch(self.runtime['epoch'], is_best=is_best)

    def before_start(self):
        """Initialization before starting training or evaluating."""
        if self.cfg.attack_mode == 'patch':
            if self.cfg.patch.get('resume_path'):
                self.model.load_patch(self.cfg.patch.resume_path)
            self.runtime = {}  # to store some variables during the training period
            self.runtime['total_loss'] = {'value': 0, 'length': 0}
            self.runtime['epoch'] = 1
            self.runtime['lowest_bbox_mAP'] = 1.0
        self.is_distributed = is_distributed()
        self.rank = 0 if not self.is_distributed else dist.get_rank()
        self.device = torch.device(self.rank)
        self.world_size = get_word_size() if self.is_distributed else 1
        self.model = self.model.to(self.device)
        self.model.detector.to(self.device)
        if self.is_distributed:
            self.model = DistributedDataParallel(self.model, device_ids=[self.device], output_device=self.device,
                                                 find_unused_parameters=True,
                                                 )
            self.model.module.freeze_layers(self.model.module.detector)
        else:
            self.model.freeze_layers(self.model.detector)

    def scale_lr(self):
        '''Automatically scale learning rate based on base batch size and real batch size'''
        if self.cfg.get('auto_lr_scaler'):
            base_batch_size = self.cfg.auto_lr_scaler.base_batch_size
            real_batch_size = self.world_size * self.cfg.batch_size
            ratio = float(real_batch_size) / float(base_batch_size)
            self.cfg.optimizer.kwargs.lr *= ratio
