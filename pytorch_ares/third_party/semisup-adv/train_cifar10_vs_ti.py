"""
Train data sourcing model. Based on code from
https://github.com/hysts/pytorch_shake_shake
"""
import argparse
from collections import OrderedDict
import importlib
import json
import logging
import pathlib
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torchvision

from utils import get_model

import pdb

from dataloader import get_cifar10_vs_ti_loader

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

global_step = 0


def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def parse_args():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument('--model', type=str, default='wrn-28-10')
    
    # run config
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_freq', type=int, default=20)

    # optim config
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--base_lr', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--lr_min', type=float, default=0)


    args = parser.parse_args()

    # 10 CIFAR10 classes and one non-CIFAR10 class
    model_config = OrderedDict([
        ('name', args.model),
        ('n_classes', 11),
    ])

    optim_config = OrderedDict([
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        ('base_lr', args.base_lr),
        ('weight_decay', args.weight_decay),
        ('momentum', args.momentum),
        ('nesterov', args.nesterov),
        ('lr_min', args.lr_min),
        ('cifar10_fraction', 0.5)
    ])

    data_config = OrderedDict([
        ('dataset', 'CIFAR10VsTinyImages'),
        ('dataset_dir', args.data_dir),
    ])

    run_config = OrderedDict([
        ('seed', args.seed),
        ('outdir', args.output_dir),
        ('num_workers', args.num_workers),
        ('device', args.device),
        ('save_freq', args.save_freq),
    ])

    config = OrderedDict([
        ('model_config', model_config),
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config),
    ])

    return config


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def _cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
        1 + np.cos(step / total_steps * np.pi))


def get_cosine_annealing_scheduler(optimizer, optim_config):
    total_steps = optim_config['epochs'] * optim_config['steps_per_epoch']

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_annealing(
            step,
            total_steps,
            1,  # since lr_lambda computes multiplicative factor
            optim_config['lr_min'] / optim_config['base_lr']))

    return scheduler


def train(epoch, model, optimizer, scheduler, criterion, train_loader,
          run_config):
    global global_step

    logger.info('Train {}'.format(epoch))

    model.train()
    device = torch.device(run_config['device'])

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    accuracy_c10_meter = AverageMeter()
    accuracy_c10_v_ti_meter = AverageMeter()
    start = time.time()
    
    for step, (data, targets) in enumerate(train_loader):
        global_step += 1


        scheduler.step()

        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.item()
        correct_ = preds.eq(targets).sum().item()
        num = data.size(0)

        accuracy = correct_ / num

        loss_meter.update(loss_, num)
        accuracy_meter.update(accuracy, num)

        is_c10 = targets != 10
        num_c10 = is_c10.float().sum().item()
        # Computing cifar10 accuracy
        if num_c10 > 0:
            _, preds_c10 = torch.max(outputs[is_c10, :10], dim=1)
            correct_c10_ = preds_c10.eq(targets[is_c10]).sum().item()
            accuracy_c10_meter.update(correct_c10_ / num_c10, num_c10)

        # Computing cifar10 vs. ti accuracy
        correct_c10_v_ti_ = (preds != 10).float().eq(
            is_c10.float()).sum().item()
        accuracy_c10_v_ti_meter.update(correct_c10_v_ti_ / num, num)


        if step % 100 == 0:
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'Accuracy {:.4f} ({:.4f}) '
                        'C10 Acc {:.4f} ({:.4f}) '
                        'Vs Acc {:.4f} ({:.4f})'.format(
                epoch,
                step,
                len(train_loader),
                loss_meter.val,
                loss_meter.avg,
                accuracy_meter.val,
                accuracy_meter.avg,
                accuracy_c10_meter.val,
                accuracy_c10_meter.avg,
                accuracy_c10_v_ti_meter.val,
                accuracy_c10_v_ti_meter.avg
            ))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    train_log = OrderedDict({
        'epoch':
            epoch,
        'train':
            OrderedDict({
                'loss': loss_meter.avg,
                'accuracy': accuracy_meter.avg,
                'accuracy_c10': accuracy_c10_meter.avg,
                'accuracy_vs': accuracy_c10_v_ti_meter.avg,
                'time': elapsed,
            }),
    })
    return train_log


def test(epoch, model, criterion, test_loader, run_config):
    logger.info('Test {}'.format(epoch))

    model.eval()
    device = torch.device(run_config['device'])

    loss_meter = AverageMeter()
    correct_c10_meter = AverageMeter()
    correct_c10_v_ti_meter = AverageMeter()
    start = time.time()
    with torch.no_grad():
        for step, (data, targets) in enumerate(test_loader):
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            _, preds = torch.max(outputs, dim=1)
            loss_ = loss.item()
            num = data.size(0)
            loss_meter.update(loss_, num)

            is_c10 = targets != 10
            # cifar10 accuracy 
            if is_c10.float().sum() > 0:
                _, preds_c10 = torch.max(outputs[is_c10, :10], dim=1)
                correct_c10_ = preds_c10.eq(targets[is_c10]).sum().item()
                correct_c10_meter.update(correct_c10_, 1)
                
            # cifar10 vs. TI accuracy
            correct_c10_v_ti_ = (preds != 10).float().eq(
                is_c10.float()).sum().item()
            correct_c10_v_ti_meter.update(correct_c10_v_ti_, 1)

    test_targets = np.array(test_loader.dataset.targets)
    accuracy_c10 = (correct_c10_meter.sum /
                    (test_targets < 10).sum())
    accuracy_vs = (correct_c10_v_ti_meter.sum / len(test_targets))

    logger.info('Epoch {} Loss {:.4f} Accuracy inside C10 {:.4f},'
                ' C10-vs-TI {:.4f}'.format(
        epoch, loss_meter.avg, accuracy_c10, accuracy_vs))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))


    test_log = OrderedDict({
        'epoch':
            epoch,
        'test':
            OrderedDict({
                'loss': loss_meter.avg,
                'accuracy_c10': accuracy_c10,
                'accuracy_vs': accuracy_vs,
                'time': elapsed,
            }),
    })
    return test_log


def main():
    # parse command line arguments
    config = parse_args()
    logger.info(json.dumps(config, indent=2))

    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']

    # set random seed
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create output directory
    outdir = pathlib.Path(run_config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)
    save_freq = run_config['save_freq']

    # save config as json file in output directory


    outpath = outdir / 'config.json'
    with open(outpath, 'w') as fout:
        json.dump(config, fout, indent=2)

    # data loaders
    train_loader, test_loader = get_cifar10_vs_ti_loader(
        optim_config['batch_size'],
        run_config['num_workers'],
        run_config['device'] != 'cpu',
        optim_config['cifar10_fraction'],
        dataset_dir=data_config['dataset_dir'], 
        logger=logger)
    
    logger.info('Instantiated data loaders')
    # model
    model = get_model(config['model_config']['name'],
                      num_classes=config['model_config']['n_classes'],
                      normalize_input=False)
    model = torch.nn.DataParallel(model.cuda())
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info('n_params: {}'.format(n_params))

    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    weight=torch.Tensor(
                                        [1] * 10 + [0.1])).cuda()

    # optimizer
    optim_config['steps_per_epoch'] = len(train_loader)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=optim_config['base_lr'],
        momentum=optim_config['momentum'],
        weight_decay=optim_config['weight_decay'],
        nesterov=optim_config['nesterov'])
    scheduler = get_cosine_annealing_scheduler(optimizer, optim_config)

    # run test before start training
    test(0, model, criterion, test_loader, run_config)

    epoch_logs = []
    for epoch in range(1, optim_config['epochs'] + 1):
        train_log = train(epoch, model, optimizer, scheduler, criterion,
                          train_loader, run_config)
        test_log = test(epoch, model, criterion, test_loader, run_config)

        epoch_log = train_log.copy()
        epoch_log.update(test_log)
        epoch_logs.append(epoch_log)
        with open(outdir / 'log.json', 'w') as fout:
            json.dump(epoch_logs, fout, indent=2)

        if epoch % save_freq == 0 or epoch == optim_config['epochs']:
            state = OrderedDict([
                ('config', config),
                ('state_dict', model.state_dict()),
                ('optimizer', optimizer.state_dict()),
                ('epoch', epoch),
                ('accuracy_vs', test_log['test']['accuracy_vs']),
            ])
            model_path = outdir / ('model_state_epoch%d.pth' % epoch)
            torch.save(state, model_path)


if __name__ == '__main__':
    main()
