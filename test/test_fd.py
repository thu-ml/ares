#!/usr/bin/env python3
import os
import tensorflow as tf
import numpy as np

from realsafe import CrossEntropyLoss, BIM
from realsafe.model.loader import load_model_from_path
from realsafe.dataset import imagenet, dataset_to_iterator

batch_size = 25

session = tf.Session()

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../example/imagenet/resnet152_fd.py')
rs_model = load_model_from_path(model_path)
model = rs_model.load(session)

xs_ph = tf.placeholder(model.x_dtype, shape=(batch_size, *model.x_shape))
lgs, lbs = model.logits_and_labels(xs_ph)

dataset = imagenet.load_dataset_for_classifier(model, load_target=True)
dataset = dataset.batch(batch_size).take(10)

loss = CrossEntropyLoss(model)
attack = BIM(
    model=model,
    batch_size=batch_size,
    loss=loss,
    goal='ut',
    distance_metric='l_inf',
    session=session
)
attack.config(
    iteration=50,
    magnitude=8.0 / 255.0,
    alpha=0.5 / 255.0,
)


accs, adv_accs = [], []
for filenames, xs, ys, ys_target in dataset_to_iterator(dataset, session):
    xs_adv = attack.batch_attack(xs, ys=ys)

    lbs_pred = session.run(lbs, feed_dict={xs_ph: xs})
    lbs_adv = session.run(lbs, feed_dict={xs_ph: xs_adv})

    accs.append(np.equal(ys, lbs_pred).astype(np.float).mean())
    adv_accs.append(np.equal(ys, lbs_adv).astype(np.float).mean())
    print(accs[-1], adv_accs[-1])

print(np.mean(accs), np.mean(adv_accs))