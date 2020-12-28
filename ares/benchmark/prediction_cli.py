''' Run predictions with a classifier and save results to a file. '''

if __name__ == '__main__':
    import argparse

    import numpy as np
    import tensorflow as tf

    from ares.model.loader import load_model_from_path
    from ares.dataset import dataset_to_iterator

    PARSER = argparse.ArgumentParser(description='Run predictions with a classifier.')

    PARSER.add_argument('--dataset', help='Dataset for this model.', choices=['cifar10', 'imagenet'], required=True)
    PARSER.add_argument('--offset', help='Dataset offset.', type=int, required=True)
    PARSER.add_argument('--count', help='Number of examples to predict.', type=int, required=True)
    PARSER.add_argument('--output', help='Path to save prediction result.', type=str, required=True)

    PARSER.add_argument('model', help='Path to the model\'s python source file.')

    PARSER.add_argument('--batch-size', help='Batch size for running prediction.', type=int, required=True)

    args = PARSER.parse_args()

    logger = tf.get_logger()
    logger.setLevel(tf.logging.INFO)

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

    print('Running prediction...')
    batch_size = args.batch_size
    xs_ph = tf.placeholder(model.x_dtype, shape=(batch_size, *model.x_shape))
    labels = model.labels(xs_ph)
    rs = {'ys': [], 'ys_target': [], 'predictions': []}
    for i_batch, (_, xs, ys, ys_target) in enumerate(dataset_to_iterator(dataset.batch(batch_size), session)):
        rs['ys'].append(ys)
        rs['ys_target'].append(ys_target)
        predictions = session.run(labels, feed_dict={xs_ph: xs})
        rs['predictions'].append(predictions)
        acc = np.equal(predictions, ys).astype(np.float32).mean()
        logger.info('n={}..{} acc={:3f}'.format(i_batch * batch_size, i_batch * batch_size + batch_size - 1, acc))
    rs['ys'] = np.concatenate(rs['ys'])
    rs['ys_target'] = np.concatenate(rs['ys_target'])
    rs['predictions'] = np.concatenate(rs['predictions'])
    logger.info('acc={:3f}'.format((rs['predictions'] == rs['ys']).astype(np.float).mean()))

    print('Saving results...')
    np.save(args.output, rs)
