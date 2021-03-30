import argparse
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ares.dataset import cifar10, imagenet, dataset_to_iterator
from ares.model import load_model_from_path

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "example")
TOTAL_SIZE = 1000
BATCH_SIZE = 50


def main(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    attack_names = [x for x in args.attacks.split(",") if len(x) > 0]
    model_name = args.model
    output_directory = os.path.abspath(args.output)
    attackers = []
    for attack_name in attack_names:
        attackers.append((attack_name, __import__(attack_name).Attacker))
    run_one_model(model_name, attackers, session, output_directory)


def run_one_model(model_name, attackers, session, output_directory):
    is_imagenet = model_name.startswith("imagenet-")
    is_cifar10 = model_name.startswith("cifar10-")
    if is_imagenet:
        dataset_name = "imagenet"
        eps = 4.0 / 255.0
        model_path = os.path.join("imagenet", model_name[len("imagenet-"):] + ".py")
    elif is_cifar10:
        dataset_name = "cifar10"
        eps = 8.0 / 255.0
        model_path = os.path.join("cifar10", model_name[len("cifar10-"):] + ".py")
    else:
        raise AssertionError("Invalid model {}!".format(model_name))
    model_path = os.path.join(MODEL_PATH, model_path)

    model = load_model_from_path(model_path).load(session)
    if is_imagenet:
        dataset = imagenet.load_dataset_for_classifier(model, offset=0, load_target=False).take(TOTAL_SIZE)
    else:
        dataset = cifar10.load_dataset_for_classifier(model, offset=0, load_target=False).take(TOTAL_SIZE)
    xs_ph = tf.placeholder(model.x_dtype, shape=(None, *model.x_shape))
    labels_op = model.labels(xs_ph)

    for attack_name, attacker_class in attackers:
        print("Running {} on {}".format(attack_name, model_name))
        attacker = attacker_class(model, BATCH_SIZE, dataset_name, session)
        attacker.config(magnitude=eps * (model.x_max - model.x_min))
        success_count = 0
        for batch, (_, xs, ys) in enumerate(tqdm(dataset_to_iterator(dataset.batch(BATCH_SIZE), session), total=TOTAL_SIZE // BATCH_SIZE)):
            xs_adv = attacker.batch_attack(xs.copy(), ys=ys.copy()).astype(np.float32)
            xs_adv = np.clip(xs_adv, xs - eps * (model.x_max - model.x_min), xs + eps * (model.x_max - model.x_min))
            xs_adv = np.clip(xs_adv, model.x_min, model.x_max)
            assert not np.any(np.isnan(xs_adv))
            labels = session.run(labels_op, feed_dict={xs_ph: xs_adv})
            success_count += np.sum(np.logical_not(np.equal(labels, ys)))
        score = success_count / TOTAL_SIZE
        print("Score for {} on {}: {}".format(attack_name, model_name, score))
        with open(os.path.join(output_directory, "{}.csv".format(attack_name)), "a") as f:
            f.write("{},{}\n".format(model_name, score))
            f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--attacks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args)
