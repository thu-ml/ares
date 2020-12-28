''' Provide a command line tool to call DistortionBenchmark directly.

The output file contains metadata of the benchmark (including full cmdline) and its result in a dictionary. To load the
result:

.. code-block:: python

   import numpy as np
   result = np.load('path/to/output.npy', allow_pickle=True).item()

The result is an array of the minimal distortion value found by the attack method for each input. If the attack method
failed to generate an adversarial example, the value is set to ``np.nan``.
'''

if __name__ == '__main__':
    import sys
    import argparse
    import numpy as np
    import tensorflow as tf

    from ares import CrossEntropyLoss, CWLoss
    from ares.model.loader import load_model_from_path
    from ares.benchmark.distortion import DistortionBenchmark

    PARSER = argparse.ArgumentParser(description='Run distortion benchmark on a classifier.')

    PARSER.add_argument(
        '--method', help='Attack method.', required=True,
        choices=['fgsm', 'bim', 'pgd', 'mim', 'cw', 'deepfool', 'nes', 'spsa', 'nattack'],
    )
    PARSER.add_argument('--dataset', help='Dataset for this model.', choices=['cifar10', 'imagenet'], required=True)
    PARSER.add_argument('--offset', help='Dataset offset.', type=int, required=True)
    PARSER.add_argument('--count', help='Number of examples to attack.', type=int, required=True)
    PARSER.add_argument('--output', help='Path to save benchmark result.', type=str, required=True)
    PARSER.add_argument('--iteration', type=int, help='Iteration parameter for the benchmark on white-box attacks.')
    PARSER.add_argument('--max-queries', type=int, help='Max queries for the benchmark on black-box attacks.')

    PARSER.add_argument('model', help='Path to the model\'s python source file.')

    PARSER.add_argument('--distortion', type=float, help='Starting point of distortion value for searching.')
    PARSER.add_argument('--confidence', type=float, default=0.0)
    PARSER.add_argument('--search-steps', type=int, default=5)
    PARSER.add_argument('--binsearch-steps', type=int, default=10)
    PARSER.add_argument('--nes-lr-factor', type=float, default=0.15)
    PARSER.add_argument('--nes-min-lr-factor', type=float, default=0.015)
    # spsa's learning rate needs to be adjusted according to distance metrics, no default value here.
    PARSER.add_argument('--spsa-lr-factor', type=float)

    # Attack method initialization parameters
    PARSER.add_argument('--goal', help='Attack goal.', required=True, choices=['t', 'tm', 'ut'])
    PARSER.add_argument('--distance-metric', help='Attack\' distance metric.', required=True, choices=['l_2', 'l_inf'])
    PARSER.add_argument('--batch-size', help='Batch size hint for attack method.', type=int, required=True)

    PARSER.add_argument('--cw-loss-c', help='C&W attack\'s c parameter.', type=float)
    PARSER.add_argument('--learning-rate', help='C&W attack\'s learning_rate parameter.', type=float)
    PARSER.add_argument('--init-distortion', help='NAttack\'s init_distortion parameter.', type=float)
    PARSER.add_argument('--samples-per-draw', type=int)
    PARSER.add_argument('--dimension-reduction-height', type=int)
    PARSER.add_argument('--dimension-reduction-width', type=int)

    PARSER.add_argument('--rand-init-magnitude', type=float)
    PARSER.add_argument('--decay-factor', type=float)
    PARSER.add_argument('--cs', type=float)
    PARSER.add_argument('--overshot', type=float)
    PARSER.add_argument('--sigma', type=float)
    PARSER.add_argument('--lr-tuning', action='store_true', default=False)
    PARSER.add_argument('--plateau-length', type=int)
    PARSER.add_argument('--beta1', type=float)
    PARSER.add_argument('--beta2', type=float)
    PARSER.add_argument('--epsilon', type=float)
    PARSER.add_argument('--lr', type=float, help='NAttack\'s learning rate parameter.')

    PARSER.add_argument('--logger', action='store_true', default=False)

    args = PARSER.parse_args()

    distortion = None
    if args.method in ('fgsm', 'bim', 'pgd', 'mim', 'nes', 'spsa', 'nattack'):
        distortion = args.distortion
        if distortion is None:
            PARSER.error('{} attack require the --distortion parameter.'.format(args.method))

    config_kwargs = dict()
    for kwarg in ('iteration', 'max_queries', 'rand_init_magnitude', 'decay_factor', 'cs', 'overshot', 'sigma',
                  'lr_tuning', 'plateau_length', 'beta1', 'beta2', 'epsilon', 'lr'):
        attr = getattr(args, kwarg)
        if attr is not None:
            config_kwargs[kwarg] = attr
    if args.method == 'cw':
        config_kwargs['search_steps'] = args.search_steps
        config_kwargs['binsearch_steps'] = args.binsearch_steps

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

    # the attack's __init__() parameters
    kwargs = dict()
    for kwarg in ('cw_loss_c', 'learning_rate', 'init_distortion', 'samples_per_draw'):
        attr = getattr(args, kwarg)
        if attr is not None:
            kwargs[kwarg] = attr
    if args.dimension_reduction_height is not None and args.dimension_reduction_width is not None:
        kwargs['dimension_reduction'] = (args.dimension_reduction_height, args.dimension_reduction_width)
    if attack_name in ('fgsm', 'bim', 'pgd', 'mim'):
        kwargs['loss'] = CrossEntropyLoss(model)
    elif attack_name in ('nes', 'spsa', 'nattack'):
        kwargs['loss'] = CWLoss(model)

    benchmark = DistortionBenchmark(attack_name, model, batch_size, goal, distance_metric, session,
                                    distortion, args.confidence, args.search_steps, args.binsearch_steps,
                                    args.nes_lr_factor, args.nes_min_lr_factor, args.spsa_lr_factor,
                                    **kwargs)

    print('Configuring attack...')
    benchmark.config(**config_kwargs)

    print('Running benchmark...')
    rs = benchmark.run(dataset, logger)

    print('Saving benchmark result...')
    np.save(args.output, {
        'type': 'distortion',
        'method': attack_name,
        'goal': goal,
        'distance_metric': distance_metric,
        'cmdline': ' '.join(sys.argv[:]),
        'result': rs
    })
