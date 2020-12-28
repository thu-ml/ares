''' Provide a command line tool to call AttackBenchmark directly. '''

if __name__ == '__main__':
    import argparse
    import numpy as np
    import tensorflow as tf

    from ares import CrossEntropyLoss, CWLoss
    from ares.model.loader import load_model_from_path
    from ares.benchmark.attack import AttackBenchmark

    PARSER = argparse.ArgumentParser(description='Run attack on a classifier.')

    PARSER.add_argument(
        '--method', help='Attack method.', required=True,
        choices=['fgsm', 'bim', 'pgd', 'mim', 'cw', 'deepfool', 'nes', 'spsa', 'nattack', 'boundary', 'evolutionary'],
    )
    PARSER.add_argument('--dataset', help='Dataset for this model.', choices=['cifar10', 'imagenet'], required=True)
    PARSER.add_argument('--offset', help='Dataset offset.', type=int, required=True)
    PARSER.add_argument('--count', help='Number of examples to attack.', type=int, required=True)

    PARSER.add_argument('model', help='Path to the model\'s python source file.')

    # Attack method initialization parameters
    PARSER.add_argument('--goal', help='Attack goal.', required=True, choices=['t', 'tm', 'ut'])
    PARSER.add_argument('--distance-metric', help='Attack\' distance metric.', required=True, choices=['l_2', 'l_inf'])
    PARSER.add_argument('--batch-size', help='Batch size hint for attack method.', type=int, required=True)
    PARSER.add_argument('--learning-rate', help='Learning rate in CW attack.', type=float)
    PARSER.add_argument('--cw-loss-c', help='CWLoss\'s c value in CW attack.', type=float)
    PARSER.add_argument('--samples-per-draw', type=int)
    PARSER.add_argument('--init-distortion', type=float)
    PARSER.add_argument('--dimension-reduction-height', type=int)
    PARSER.add_argument('--dimension-reduction-width', type=int)

    PARSER.add_argument('--iteration', type=int)
    PARSER.add_argument('--max-queries', type=int)
    PARSER.add_argument('--magnitude', type=float)
    PARSER.add_argument('--alpha', type=float)
    PARSER.add_argument('--rand-init-magnitude', type=float)
    PARSER.add_argument('--decay-factor', type=float)
    PARSER.add_argument('--cs', type=float)
    PARSER.add_argument('--search-steps', type=int)
    PARSER.add_argument('--binsearch-steps', type=int)
    PARSER.add_argument('--overshot', type=float)
    PARSER.add_argument('--sigma', type=float)
    PARSER.add_argument('--lr', type=float)
    PARSER.add_argument('--min-lr', type=float)
    PARSER.add_argument('--lr-tuning', action='store_true', default=False)
    PARSER.add_argument('--plateau-length', type=int)
    PARSER.add_argument('--max-directions', type=int)
    PARSER.add_argument('--spherical-step', type=float)
    PARSER.add_argument('--source-step', type=float)
    PARSER.add_argument('--step-adaptation', type=float)
    PARSER.add_argument('--mu', type=float)
    PARSER.add_argument('--c', type=float)
    PARSER.add_argument('--maxprocs', type=int)
    PARSER.add_argument('--logger', action='store_true', default=False)

    args = PARSER.parse_args()

    config_kwargs = dict()
    for kwarg in ('iteration', 'max_queries', 'magnitude', 'alpha', 'rand_init_magnitude', 'decay_factor', 'cs',
                  'search_steps', 'binsearch_steps', 'overshot', 'sigma', 'lr', 'min_lr', 'lr_tuning', 'plateau_length',
                  'max_directions', 'spherical_step', 'source_step', 'step_adaptation', 'mu', 'c', 'maxprocs'):
        attr = getattr(args, kwarg)
        if attr is not None:
            config_kwargs[kwarg] = attr

    logger = tf.get_logger()
    logger.setLevel(tf.logging.INFO)
    if args.logger:
        config_kwargs['logger'] = logger

    print('Loading tensorflow session...')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    print('Loading model...')
    model = load_model_from_path(args.model).load(session)

    print('Loading dataset...')
    if args.dataset == 'cifar10':
        from ares.dataset import cifar10
        dataset = cifar10.load_dataset_for_classifier(model, offset=args.offset, load_target=True)
    else:
        from ares.dataset import imagenet
        dataset = imagenet.load_dataset_for_classifier(model, offset=args.offset, load_target=True)
    dataset = dataset.take(args.count)

    print('Loading attack...')
    attack_name, batch_size, dataset_name = args.method, args.batch_size, args.dataset
    goal, distance_metric = args.goal, args.distance_metric

    kwargs = dict()
    for kwarg in ('learning_rate', 'cw_loss_c', 'samples_per_draw', 'init_distortion'):
        attr = getattr(args, kwarg)
        if attr is not None:
            kwargs[kwarg] = attr
    if args.dimension_reduction_height is not None and args.dimension_reduction_width is not None:
        kwargs['dimension_reduction'] = (args.dimension_reduction_height, args.dimension_reduction_width)
    if attack_name in ('fgsm', 'bim', 'pgd', 'mim'):
        kwargs['loss'] = CrossEntropyLoss(model)
    elif attack_name in ('nes', 'spsa', 'nattack'):
        kwargs['loss'] = CWLoss(model)

    benchmark = AttackBenchmark(attack_name, model, batch_size, dataset_name, goal, distance_metric, session, **kwargs)

    print('Configuring attack...')
    benchmark.config(**config_kwargs)

    print('Running benchmark...')
    acc, acc_adv, total, succ, dist = benchmark.run(dataset, logger)
    print('n={}, acc={:3f}, adv_acc={:3f}, succ={:3f}, dist_mean={:3f}'.format(
        args.count,
        np.mean(acc.astype(np.float)), np.mean(acc_adv.astype(np.float)),
        np.sum(succ.astype(np.float)) / np.sum(total.astype(np.float)), np.mean(dist)
    ))
