from __future__ import print_function
from torch.optim import lr_scheduler
from ares.utils.registry import Registry
from ares.utils.logger import format_print
@Registry.register_lr_scheduler()
class ALRS:
    """Reference:Bootstrap Generalization Ability from Loss Landscape Perspective."""

    def __init__(self, optimizer, loss_threshold=1e-4, loss_ratio_threshold=1e-4, decay_rate=0.97, patience=10,
                 last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.loss_threshold = loss_threshold
        self.decay_rate = decay_rate
        self.loss_ratio_threshold = loss_ratio_threshold
        self.last_loss = 999
        self.total_epoch_loss = 0
        self.patience = patience
        self.last_epoch = last_epoch
        self.verbose = verbose

    def update_lr(self, loss):
        delta = self.last_loss - loss
        if delta < self.loss_threshold and delta / self.last_loss < self.loss_ratio_threshold:
            for ind, group in enumerate(self.optimizer.param_groups):
                self.optimizer.param_groups[ind]['lr'] *= self.decay_rate
                now_lr = group['lr']
                if self.verbose:
                    print(f'now lr = {now_lr}')

    @format_print()
    def step(self, loss, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        if self.last_epoch % self.patience != 0:
            self.total_epoch_loss += loss
        else:
            loss = self.total_epoch_loss / self.patience
            self.update_lr(loss)
            self.last_loss = loss
            self.total_epoch_loss = 0

@Registry.register_lr_scheduler()
class warmupALRS(ALRS):
    """Reference:Bootstrap Generalization Ability from Loss Landscape Perspective"""

    def __init__(self, optimizer, warmup_epoch=50, loss_threshold=1e-4, loss_ratio_threshold=1e-4, decay_rate=0.97, last_epoch=-1, verbose=False):
        super().__init__(optimizer, loss_threshold, loss_ratio_threshold, decay_rate, last_epoch, verbose)
        self.warmup_rate = 1 / 3
        self.warmup_epoch = warmup_epoch
        self.start_lr = optimizer.param_groups[0]["lr"]
        self.warmup_lr = self.start_lr * (1 - self.warmup_rate)
        self.update_lr(lambda x: x * self.warmup_rate)

    def update_lr(self, update_fn):
        for ind, group in enumerate(self.optimizer.param_groups):
            self.optimizer.param_groups[ind]['lr'] = update_fn(group['lr'])
            now_lr = group['lr']
            if self.verbose:
                print(f'now lr = {now_lr}')

    @format_print()
    def step(self, loss, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        if self.last_epoch < self.warmup_epoch:
            self.update_lr(lambda x: -(self.warmup_epoch - epoch) * self.warmup_lr / self.warmup_epoch + self.start_lr)
        elif self.last_epoch % self.patience != 0:
            self.total_epoch_loss += loss
        else:
            loss = self.total_epoch_loss / self.patience
            delta = self.last_loss - loss
            self.last_loss = loss
            if delta < self.loss_threshold and delta / self.last_loss < self.loss_ratio_threshold:
                self.update_lr(lambda x: x * self.decay_rate)

@Registry.register_lr_scheduler()
class CosineLR(lr_scheduler.CosineAnnealingLR):
    '''See torch.optim.lr_scheduler.CosineAnnealingLR for details'''

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
        super().__init__(optimizer, T_max, eta_min, last_epoch, verbose)

    @format_print()
    def step(self, epoch=None, **kwargs) -> None:
        super().step(epoch)

@Registry.register_lr_scheduler()
class ExponentialLR(lr_scheduler.ExponentialLR):
    '''See torch.optim.lr_scheduler.ExponentialLR for details'''

    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        super().__init__(optimizer, gamma, last_epoch, verbose)

    @format_print()
    def step(self, epoch=None, **kwargs) -> None:
        super().step(epoch)

@Registry.register_lr_scheduler()
class PlateauLR(lr_scheduler.ReduceLROnPlateau):
    '''See torch.optim.lr_scheduler.ReduceLROnPlateau for details'''

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):
        super().__init__(optimizer, mode, factor, patience, threshold,
                         threshold_mode, cooldown, min_lr, eps, verbose)

    @format_print()
    def step(self, metrics, epoch=None, **kwargs) -> None:
        super().step(metrics, epoch)

@Registry.register_lr_scheduler()
class MultiStepLR(lr_scheduler.MultiStepLR):
    '''See torch.optim.lr_scheduler.MultiStepLR for details'''

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        super().__init__(optimizer, milestones, gamma, last_epoch, verbose)

    @format_print()
    def step(self, epoch=None, **kwargs) -> None:
        super().step(epoch)


def build_lr_scheduler(optimizer, **kwargs):
    '''build learning rate scheduler based on given optimizer, lr scheduler name and its arguments'''
    return Registry.get_lr_scheduler(kwargs['type'])(optimizer, **kwargs['kwargs'])
