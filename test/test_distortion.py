import tensorflow as tf
import numpy as np
import os

from realsafe import CrossEntropyLoss
from realsafe.dataset import cifar10
from realsafe.model.loader import load_model_from_path
from realsafe.benchmark.distortion import DistortionBenchmark


logger = tf.get_logger()
logger.setLevel(tf.logging.INFO)

batch_size = 500

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../example/cifar10/resnet56.py')
rs_model = load_model_from_path(model_path)
model = rs_model.load(session)

dataset = cifar10.load_dataset_for_classifier(model, load_target=True)

benchmark = DistortionBenchmark('fgsm', model, batch_size, 'cifar10', 'ut', 'l_inf', session, 0.05, 0.00,
                                search_step=5, binsearch_step=10, loss=CrossEntropyLoss(model))
rs = benchmark.run(dataset, logger)
print((rs[np.logical_not(np.isnan(rs))] < 0.05).astype(np.float).sum() / len(rs))