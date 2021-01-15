import os
import time
import logging

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
EXAMPLE_PATH = os.path.join(BASE_PATH, "example")

LOG_BASE_PATH = "/var/log/attacker"
TASK_LOG_PATH = os.path.join(LOG_BASE_PATH, "task")

TOTAL_TIME = float(60 * 60 * 3)
STATUS_NO_FINISH = 0
STATUS_SUCCESS = 1
STATUS_FILED = 2

model_parameters = {
    # cifar10
    "cifar10—pgd_at": {
        'model_path': os.path.join(EXAMPLE_PATH, 'cifar10/pgd_at.py'),
        "batch_size": 100,
        "dataset_size": 1000,
        "max_forward": 200 * 1000,
        "max_backward": 100 * 1000,
        "dataset": "cifar10",
        "output_path": "/home/contest/results",
    },
    "cifar10-wideresnet_trades": {
        "model_path": os.path.join(EXAMPLE_PATH, 'cifar10/wideresnet_trades.py'),
        "batch_size": 100,
        "dataset_size": 1000,
        "max_forward": 200 * 1000,
        "max_backward": 100 * 1000,
        "dataset": "cifar10",
        "output_path": "/home/contest/results"
    },
    "cifar10-feature_scatter": {
        'model_path': os.path.join(EXAMPLE_PATH, 'cifar10/feature_scatter.py'),
        "batch_size": 100,
        "dataset_size": 1000,
        "max_forward": 200 * 1000,
        "max_backward": 100 * 1000,
        "dataset": "cifar10",
        "output_path": "/home/contest/results"
    },
    "cifar10-robust_overfitting": {
        'model_path': os.path.join(EXAMPLE_PATH, 'cifar10/robust_overfitting.py'),
        "batch_size": 100,
        "dataset_size": 1000,
        "max_forward": 200 * 1000,
        "max_backward": 100 * 1000,
        "dataset": "cifar10",
        "output_path": "/home/contest/results"
    },
    "cifar10-rst": {
        'model_path': os.path.join(EXAMPLE_PATH, 'cifar10/rst.py'),
        "batch_size": 100,
        "dataset_size": 1000,
        "max_forward": 200 * 1000,
        "max_backward": 100 * 1000,
        "dataset": "cifar10",
        "output_path": "/home/contest/results"
    },
    "cifar10-fast_at": {
        'model_path': os.path.join(EXAMPLE_PATH, 'cifar10/fast_at.py'),
        "batch_size": 100,
        "dataset_size": 1000,
        "max_forward": 200 * 1000,
        "max_backward": 100 * 1000,
        "dataset": "cifar10",
        "output_path": "/home/contest/results"
    },
    "cifar10-at_he": {
        'model_path': os.path.join(EXAMPLE_PATH, 'cifar10/at_he.py'),
        "batch_size": 100,
        "dataset_size": 1000,
        "max_forward": 200 * 1000,
        "max_backward": 100 * 1000,
        "dataset": "cifar10",
        "output_path": "/home/contest/results"
    },
    "cifar10-pre_training": {
        'model_path': os.path.join(EXAMPLE_PATH, 'cifar10/pre_training.py'),
        "batch_size": 100,
        "dataset_size": 1000,
        "max_forward": 200 * 1000,
        "max_backward": 100 * 1000,
        "dataset": "cifar10",
        "output_path": "/home/contest/results"
    },
    "cifar10-mmc": {
        'model_path': os.path.join(EXAMPLE_PATH, 'cifar10/mmc.py'),
        "batch_size": 100,
        "dataset_size": 1000,
        "max_forward": 200 * 1000,
        "max_backward": 100 * 1000,
        "dataset": "cifar10",
        "output_path": "/home/contest/results"
    },
    "cifar10-free_at": {
        'model_path': os.path.join(EXAMPLE_PATH, 'cifar10/free_at.py'),
        "batch_size": 100,
        "dataset_size": 1000,
        "max_forward": 200 * 1000,
        "max_backward": 100 * 1000,
        "dataset": "cifar10",
        "output_path": "/home/contest/results"
    },
    "cifar10-awp": {
        'model_path': os.path.join(EXAMPLE_PATH, 'cifar10/awp.py'),
        "batch_size": 100,
        "dataset_size": 1000,
        "max_forward": 200 * 1000,
        "max_backward": 100 * 1000,
        "dataset": "cifar10",
        "output_path": "/home/contest/results"
    },
    "cifar10-hydra": {
        'model_path': os.path.join(EXAMPLE_PATH, 'cifar10/hydra.py'),
        "batch_size": 100,
        "dataset_size": 1000,
        "max_forward": 200 * 1000,
        "max_backward": 100 * 1000,
        "dataset": "cifar10",
        "output_path": "/home/contest/results"
    },
    "cifar10-label_smoothing": {
        'model_path': os.path.join(EXAMPLE_PATH, 'cifar10/label_smoothing.py'),
        "batch_size": 100,
        "dataset_size": 1000,
        "max_forward": 200 * 1000,
        "max_backward": 100 * 1000,
        "dataset": "cifar10",
        "output_path": "/home/contest/results"
    },

    # imagenet
    "imagenet-fast_at": {
        'model_path': os.path.join(EXAMPLE_PATH, 'imagenet/fast_at.py'),
        "batch_size": 20,
        "dataset_size": 1000,
        "max_forward": 200 * 1000,
        "max_backward": 100 * 1000,
        "dataset": "imagenet",
        "output_path": "/home/contest/results"
    },
    "imagenet-free_at": {
        'model_path': os.path.join(EXAMPLE_PATH, 'imagenet/free_at.py'),
        "batch_size": 20,
        "dataset_size": 1000,
        "max_forward": 200 * 1000,
        "max_backward": 100 * 1000,
        "dataset": "imagenet",
        "output_path": "/home/contest/results"
    }
}
