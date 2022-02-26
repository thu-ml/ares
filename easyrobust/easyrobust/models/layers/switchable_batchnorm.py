import torch
import torch.nn as nn

class SwitchableBatchNorm2d(torch.nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.bn_mode = 'clean'
        self.bn_adv = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: torch.Tensor):
        if self.training:  # aux BN only relevant while training
            if self.bn_mode == 'clean':
                return super().forward(input)
            elif self.bn_mode == 'adv':
                return self.bn_adv(input)
        else:
            return super().forward(input)

def convert_switchablebn_model(module):
    """
    Recursively traverse module and its children to replace all instances of
    ``torch.nn.modules.batchnorm._BatchNorm`` with `SplitBatchnorm2d`.
    Args:
        module (torch.nn.Module): input module
        num_splits: number of separate batchnorm layers to split input across
    Example::
        >>> # model is an instance of torch.nn.Module
        >>> model = timm.models.convert_splitbn_model(model, num_splits=2)
    """
    mod = module
    if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
        return module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = SwitchableBatchNorm2d(
            module.num_features, module.eps, module.momentum, module.affine,
            module.track_running_stats)
        mod.running_mean = module.running_mean
        mod.running_var = module.running_var
        mod.num_batches_tracked = module.num_batches_tracked
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
            
        for aux in [mod.bn_adv]:
            aux.running_mean = module.running_mean.clone()
            aux.running_var = module.running_var.clone()
            aux.num_batches_tracked = module.num_batches_tracked.clone()
            if module.affine:
                aux.weight.data = module.weight.data.clone().detach()
                aux.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_switchablebn_model(child))
    del module
    return mod