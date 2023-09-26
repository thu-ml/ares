import numpy as np

attack_configs = {
    'autoattack': {'norm': np.inf, 'eps': 0.3, 'seed': None, 'verbose': False, 'attacks_to_run': [], 'version': 'standard', 'is_tf_model': False, 'logger': None},
    'bim': {'norm': np.inf, 'eps': 4/255, 'stepsize': 1/255, 'steps': 20, 'target': False, 'loss': 'ce'},
    'boundary': {'norm': 2, 'spherical_step_eps': 4/255, 'orth_step_factor': 0.5, 'perp_step_factor': 0.5, 'orthogonal_step_eps': 4/255, 'max_iter': 20, 'target': False},
    'cda': {'norm': np.inf, 'eps': 4/255, 'gk': True},
    'cw': {'norm': 2, 'target': False, 'kappa': 0.0, 'lr': 0.2, 'init_const': 0.01, 'max_iter': 4, 'binary_search_steps': 20, 'num_classes': 10},
    'deepfool': {'norm': np.inf, 'overshoot': 0.02, 'max_iter': 20, 'target': False},
    'dim': {'norm': np.inf, 'eps': 4/255, 'stepsize': 1/255, 'steps': 20, 'decay_factor': 1.0, 'resize_rate': 0.85, 'diversity_prob': 0.7, 'target': False, 'loss': 'ce'},
    'evolutionary': {'target': False, 'ccov': 0.001, 'decay_weight': 0.99, 'max_queries': 20, 'mu': 0.01, 'sigma': 3e-2, 'maxlen': 30},
    'fgsm': {'norm': np.inf, 'eps': 4/255, 'target': False, 'loss': 'ce'},
    'mim': {'norm': np.inf, 'eps': 4/255, 'stepsize': 1/255, 'steps': 20, 'decay_factor': 1.0, 'target': False, 'loss': 'ce'},
    'nattack': {'norm': np.inf, 'eps': 4/255, 'max_queries': 20, 'target': False, 'sample_size': 100, 'lr': 0.02, 'sigma': 0.1},
    'nes': {'norm': np.inf, 'eps': 4/255, 'nes_samples': 10, 'sample_per_draw': 1, 'max_queries': 20, 'stepsize': 1/255, 'search_sigma': 0.02, 'decay': 0.00, 'random_perturb_start': False, 'target': False},
    'pgd': {'norm': np.inf, 'eps': 4/255, 'stepsize': 1/255, 'steps': 20, 'target': False, 'loss': 'ce'},
    'sgm': {'net_name': 'tv_resnet50', 'norm': np.inf, 'eps': 4/255, 'stepsize': 1/255, 'steps': 20, 'gamma': 0.0, 'momentum': 1.0, 'target': False, 'loss': 'ce'},
    'si_ni_fgsm': {'norm': np.inf, 'eps': 4/255, 'scale_factor': 1, 'stepsize': 1/255, 'steps': 20, 'decay_factor': 1.0, 'target': False, 'loss': 'ce'},
    'spsa': {'norm': np.inf, 'eps': 4/255, 'learning_rate': 0.2, 'delta': 0.1, 'spsa_samples': 10, 'sample_per_draw': 10, 'nb_iter': 10, 'early_stop_loss_threshold': None, 'target': None},
    'tim': {'norm': np.inf, 'kernel_name': 'gaussian', 'len_kernel': 15, 'nsig': 3, 'eps': 4/255, 'stepsize': 1/255, 'steps': 20, 'decay_factor': 1.0, 'resize_rate': 0.85, 'diversity_prob': 0.7, 'target': False, 'loss': 'ce'},
    'tta': {'norm': np.inf, 'eps': 4/255, 'stepsize': 1/255, 'steps': 20, 'kernel_size': 5, 'nsig': 3, 'resize_rate': 0.85, 'diversity_prob': 0.7, 'target': False, 'loss': 'ce'},
    'vmi_fgsm': {'norm': np.inf, 'eps': 4/255, 'beta': 1.5, 'sample_number': 10, 'stepsize': 1/255, 'steps': 20, 'decay_factor': 1.0, 'target': False, 'loss': 'ce'},
}