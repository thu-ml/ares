"""
Evaluate robustness against specific attack.
Loosely based on code from https://github.com/yaodongyu/TRADES
"""

import os
import json
import numpy as np
import re
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from datasets import SemiSupervisedDataset, DATASETS
from torchvision import transforms
from attack_pgd import pgd
from attack_cw import cw
import torch.backends.cudnn as cudnn
from utils import get_model


def eval_adv_test(model, device, test_loader, attack, attack_params,
                  results_dir, num_eval_batches):
    """
    evaluate model by white-box attack
    """
    model.eval()
    if attack == 'pgd':
        restarts_matrices = []
        for restart in range(attack_params['num_restarts']):
            is_correct_adv_rows = []
            count = 0
            batch_num = 0
            natural_num_correct = 0
            for data, target in test_loader:
                batch_num = batch_num + 1
                if num_eval_batches and batch_num > num_eval_batches:
                    break
                data, target = data.to(device), target.to(device)
                count += len(target)
                X, y = Variable(data, requires_grad=True), Variable(target)
                # is_correct_adv has batch_size*num_iterations dimensions
                is_correct_natural, is_correct_adv = pgd(
                    model, X, y,
                    epsilon=attack_params['epsilon'],
                    num_steps=attack_params['num_steps'],
                    step_size=attack_params['step_size'],
                    random_start=attack_params['random_start'])
                natural_num_correct += is_correct_natural.sum()
                is_correct_adv_rows.append(is_correct_adv)

            is_correct_adv_matrix = np.concatenate(is_correct_adv_rows, axis=0)
            restarts_matrices.append(is_correct_adv_matrix)

            is_correct_adv_over_restarts = np.stack(restarts_matrices, axis=-1)
            num_correct_adv = is_correct_adv_over_restarts.prod(
                axis=-1).prod(axis=-1).sum()

            logging.info("Accuracy after %d restarts: %.4g%%" %
                         (restart + 1, 100 * num_correct_adv / count))
            stats = {'attack': 'pgd',
                     'count': count,
                     'attack_params': attack_params,
                     'natural_accuracy': natural_num_correct / count,
                     'is_correct_adv_array': is_correct_adv_over_restarts,
                     'robust_accuracy': num_correct_adv / count,
                     'restart_num': restart
                     }

            np.save(os.path.join(results_dir, 'pgd_results.npy'), stats)

    elif attack == 'cw':
        all_linf_distances = []
        count = 0
        for data, target in test_loader:
            logging.info('Batch: %g', count)
            count = count + 1
            if num_eval_batches and count > num_eval_batches:
                break
            data, target = data.to(device), target.to(device)
            X, y = Variable(data, requires_grad=True), Variable(target)
            batch_linf_distances = cw(model, X, y,
                                      binary_search_steps=attack_params[
                                          'binary_search_steps'],
                                      max_iterations=attack_params[
                                          'max_iterations'],
                                      learning_rate=attack_params[
                                          'learning_rate'],
                                      initial_const=attack_params[
                                          'initial_const'],
                                      tau_decrease_factor=attack_params[
                                          'tau_decrease_factor'])
            all_linf_distances.append(batch_linf_distances)
            stats = {'attack': 'cw',
                     'attack_params': attack_params,
                     'linf_distances': np.array(all_linf_distances),
                     }
            np.save(os.path.join(results_dir, 'cw_results.npy'), stats)

    else:
        raise ValueError('Unknown attack %s' % attack)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR Attack Evaluation')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=DATASETS,
                        help='The dataset')
    parser.add_argument('--model_path',
                        help='Model for attack evaluation')
    parser.add_argument('--model', '-m', default='wrn-28-10', type=str,
                        help='Name of the model')
    parser.add_argument('--output_suffix', default='', type=str,
                        help='String to add to log filename')
    parser.add_argument('--batch_size', type=int, default=200, metavar='N',
                        help='Input batch size for testing (default: 200)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--epsilon', default=0.031, type=float,
                        help='Attack perturbation magnitude')
    parser.add_argument('--attack', default='pgd', type=str,
                        help='Attack type (CW requires FoolBox)',
                        choices=('pgd', 'cw'))
    parser.add_argument('--num_steps', default=40, type=int,
                        help='Number of PGD steps')
    parser.add_argument('--step_size', default=0.01, type=float,
                        help='PGD step size')
    parser.add_argument('--num_restarts', default=5, type=int,
                        help='Number of restarts for PGD attack')
    parser.add_argument('--no_random_start', dest='random_start',
                        action='store_false',
                        help='Disable random PGD initialization')
    parser.add_argument('--binary_search_steps', default=5, type=int,
                        help='Number of binary search steps for CW attack')
    parser.add_argument('--max_iterations', default=1000, type=int,
                        help='Max number of Adam iterations in each CW'
                             ' optimization')
    parser.add_argument('--learning_rate', default=5E-3, type=float,
                        help='Learning rate for CW attack')
    parser.add_argument('--initial_const', default=1E-2, type=float,
                        help='Initial constant for CW attack')
    parser.add_argument('--tau_decrease_factor', default=0.9, type=float,
                        help='Tau decrease factor for CW attack')
    parser.add_argument('--random_seed', default=0, type=int,
                        help='Random seed for permutation of test instances')
    parser.add_argument('--num_eval_batches', default=None, type=int,
                        help='Number of batches to run evalaution on')
    parser.add_argument('--shuffle_testset', action='store_true', default=False,
                        help='Shuffles the test set')

    args = parser.parse_args()
    torch.manual_seed(args.random_seed)

    output_dir, checkpoint_name = os.path.split(args.model_path)
    epoch = int(re.search('epoch(\d+)', checkpoint_name).group(1))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir,
                                             'attack_epoch%d%s.log' %
                                             (epoch, args.output_suffix))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()

    results_dir = os.path.join(output_dir, args.output_suffix)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    logging.info('Attack evaluation')
    logging.info('Args: %s' % args)

    # settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dl_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # set up data loader
    transform_test = transforms.Compose([transforms.ToTensor(), ])
    testset = SemiSupervisedDataset(base_dataset=args.dataset,
                                    train=False, root='data',
                                    download=True,
                                    transform=transform_test)

    if args.shuffle_testset:
        np.random.seed(123)
        logging.info("Permuting testset")
        permutation = np.random.permutation(len(testset))
        testset.data = testset.data[permutation, :]
        testset.targets = [testset.targets[i] for i in permutation]

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=args.batch_size,
                                              shuffle=False, **dl_kwargs)

    checkpoint = torch.load(args.model_path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    num_classes = checkpoint.get('num_classes', 10)
    normalize_input = checkpoint.get('normalize_input', False)
    model = get_model(args.model, num_classes=num_classes,
                      normalize_input=normalize_input)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
        if not all([k.startswith('module') for k in state_dict]):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
    else:
        def strip_data_parallel(s):
            if s.startswith('module'):
                return s[len('module.'):]
            else:
                return s
        state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    attack_params = {
        'epsilon': args.epsilon,
        'seed': args.random_seed
    }
    if args.attack == 'pgd':
        attack_params.update({
            'num_restarts': args.num_restarts,
            'step_size': args.step_size,
            'num_steps': args.num_steps,
            'random_start': args.random_start,
        })
    elif args.attack == 'cw':
        attack_params.update({
            'binary_search_steps': args.binary_search_steps,
            'max_iterations': args.max_iterations,
            'learning_rate': args.learning_rate,
            'initial_const': args.initial_const,
            'tau_decrease_factor': args.tau_decrease_factor
        })

    logging.info('Running %s' % attack_params)
    eval_adv_test(model, device, test_loader, attack=args.attack,
                  attack_params=attack_params, results_dir=results_dir,
                  num_eval_batches=args.num_eval_batches)
