MEAN_ZERO=(0., 0., 0.)
STA_ONE=(1., 1., 1.)
MEAN_CIFAR=(0.4914, 0.4822, 0.4465)
STD_CIFAR=(0.2471, 0.2435, 0.2616)
MEAN_MID=(0.5, 0.5, 0.5)
STD_MID=(0.5, 0.5, 0.5)

def get_url(ckpt_name):
    url = "https://ml.cs.tsinghua.edu.cn/~xiaoyang/aresbench/ckpt-cifar10/" + ckpt_name

    return url

cifar_model_zoo={
    'at_he': {'model': 'wresnet34_10_fn', 'mean': MEAN_CIFAR, 'std': STD_CIFAR, 'url': get_url('model-wideres-pgdHE-wide10.pt'), 'pt': 'model-wideres-pgdHE-wide10.pt'},
    'awp': {'model': 'wresnet28_10', 'mean': MEAN_ZERO, 'std': STA_ONE, 'url': get_url('RST-AWP_cifar10_linf_wrn28-10.pt'), 'pt': 'RST-AWP_cifar10_linf_wrn28-10.pt'},
    'fast_at': {'model': 'preact_resnet18', 'mean': MEAN_CIFAR, 'std': STD_CIFAR, 'url': get_url('cifar_model_weights_30_epochs.pth'), 'pt': 'cifar_model_weights_30_epochs.pth'},
    'featurescatter': {'model': 'wresnet28_10', 'mean': MEAN_MID, 'std': STD_MID, 'url': get_url('checkpoint-199-ipot'), 'pt': 'checkpoint-199-ipot'},
    'hydra': {'model': 'wresnet28_10', 'mean': MEAN_ZERO, 'std': STA_ONE, 'url': get_url('model_best_dense.pth.tar'), 'pt': 'model_best_dense.pth.tar'},
    'label_smoothing': {'model': 'wresnet34_10', 'mean': MEAN_CIFAR, 'std': STD_CIFAR, 'url': get_url('model_best.pth'), 'pt': 'model_best.pth'},
    'pre_training': {'model': 'wresnet28_10', 'mean': MEAN_MID, 'std': STD_MID, 'url': get_url('cifar10wrn_baseline_epoch_4.pt'), 'pt': 'cifar10wrn_baseline_epoch_4.pt'},
    'robust_overfiting': {'model': 'wresnet34_10', 'mean': MEAN_CIFAR, 'std': STD_CIFAR, 'url': get_url('cifar10_wide10_linf_eps8.pth'), 'pt': 'cifar10_wide10_linf_eps8.pth'},
    'rst': {'model': 'wresnet28_10', 'mean': MEAN_ZERO, 'std': STA_ONE, 'url': get_url('cifar10_rst_adv.pt.ckpt'), 'pt': 'cifar10_rst_adv.pt.ckpt'},
    'trades': {'model': 'wresnet34_10', 'mean': MEAN_ZERO, 'std': STA_ONE, 'url': get_url('model_cifar_wrn.pt'), 'pt': 'model_cifar_wrn.pt'},
    
}