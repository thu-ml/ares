import os
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.distributed as dist
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.registry import MODELS
from mmengine.registry import DefaultScope
from mmengine.evaluator.evaluator import Evaluator
from ares.utils.logger import setup_logger
from ares.attack.detection.trainer import Trainer
from ares.attack.detection.attacker import UniversalAttacker
from ares.attack.detection.utils import all_reduce, mkdirs_if_not_exists
from ares.attack.detection.utils import HiddenPrints
from ares.attack.detection.utils import modify_test_pipeline
from ares.attack.detection.utils import modify_train_pipeline
# The below imports are necessary to overwrite classes in mmdet package.
from ares.attack.detection.custom import CocoDataset, CocoMetric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/demo.py', help='attack config file path')
    parser.add_argument("--eval_only", action='store_true', help="evaluate only")
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()

    attack_cfg = Config.fromfile(args.cfg)
    if args.cfg_options is not None:
        attack_cfg.merge_from_dict(args.cfg_options)
    detector_cfg = Config.fromfile(attack_cfg.detector.cfg_file)
    # modify batch size
    detector_cfg.train_dataloader.batch_size = attack_cfg.batch_size
    detector_cfg.test_dataloader.batch_size = attack_cfg.batch_size
    modify_test_pipeline(detector_cfg)
    if attack_cfg.attack_mode == 'patch' and attack_cfg.get('use_detector_train_pipeline'):
        modify_train_pipeline(detector_cfg)
    # filter out images without gt bbounding boxes. This leads to a little different detection performance on coco val2017.
    detector_cfg.test_dataloader.dataset.filter_cfg = dict(filter_empty_gt=True)
    if not detector_cfg.model.data_preprocessor.get('mean', False):
        detector_cfg.model.data_preprocessor.mean = [0.0] * 3
    if not detector_cfg.model.data_preprocessor.get('std', False):
        detector_cfg.model.data_preprocessor.std = [1.0] * 3
    if attack_cfg.get('attacked_classes', False):
        detector_cfg.test_dataloader.dataset.kept_classes = attack_cfg.attacked_classes
        detector_cfg.train_dataloader.dataset.kept_classes = attack_cfg.attacked_classes
        detector_cfg.test_evaluator.specified_classes = attack_cfg.attacked_classes
        # detector_cfg.test_evaluator.classwise = True
    else:
        if detector_cfg.test_dataloader.dataset.get('kept_classes'):
            del detector_cfg.test_dataloader.dataset.kept_classes
        if detector_cfg.train_dataloader.dataset.get('kept_classes'):
            del detector_cfg.test_dataloader.dataset.kept_classes
        if detector_cfg.test_evaluator.get('specified_classes'):
            del detector_cfg.test_evaluator.specified_classes

    device = torch.device('cpu')
    local_rank = args.local_rank
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
    if torch.cuda.is_available():
        device = torch.device(local_rank)

    # initialize detector, dataloader and evaluator
    with HiddenPrints():
        DefaultScope.get_instance('attack', scope_name='mmdet')
        detector = MODELS.build(detector_cfg.model)
        detector.eval()
        test_dataloader = Runner.build_dataloader(detector_cfg.test_dataloader, seed=0, diff_rank_seed=False)
        if attack_cfg.get('use_detector_train_pipeline'):
            train_dataloader = Runner.build_dataloader(detector_cfg.test_dataloader, seed=0, diff_rank_seed=False)#runner.train_dataloader
        else:
            train_dataloader = test_dataloader
        evaluator = Evaluator(detector_cfg.test_evaluator)
        evaluator.dataset_meta = test_dataloader.dataset.metainfo

    time_stamp = torch.tensor(time.time(), dtype=torch.float64, device=device)
    all_reduce(time_stamp, 'avg')
    time_stamp = time.strftime('%Y-%m-%d~%H:%M:%S', time.localtime(time_stamp.int().cpu().item()))
    log_dir = os.path.join('logs', os.path.splitext(os.path.basename(attack_cfg.detector.cfg_file))[0], time_stamp)
    mkdirs_if_not_exists(log_dir)
    attack_cfg.log_dir = log_dir
    if attack_cfg.get('attacked_classes', False):
        attack_cfg.attacked_labels = test_dataloader.dataset.kept_labels

    attack_cfg.all_classes = test_dataloader.dataset.metainfo['classes']
    if local_rank == 0:
        attack_cfg.dump(os.path.join(log_dir, os.path.basename(args.cfg)))
        detector_cfg.dump(os.path.join(log_dir, 'detector_cfg.py'))
    logger = setup_logger(save_dir=log_dir, distributed_rank=local_rank)
    logger.info('Relative results will be saved in %s'%log_dir)
    attacker = UniversalAttacker(attack_cfg, detector, logger, device)
    trainer = Trainer(attack_cfg, attacker, train_dataloader, test_dataloader, evaluator, logger)

    if attack_cfg.attack_mode == 'global':
        trainer.eval(eval_on_clean=True)
    elif attack_cfg.attack_mode == 'patch':
        if args.eval_only:
            assert attack_cfg.patch.resume_path, 'Adversarial patches path should not be none for eval only mode!'
            trainer.eval(eval_on_clean=True)
        else:
            trainer.train()
