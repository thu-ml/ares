#!/usr/bin/env python3
import os
import sys

BASE_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(BASE_PATH)
import pathlib
import time
import json
from mpi4py import MPI
import os
import sys
import argparse
import shutil
import numpy as np
import tensorflow as tf
from ares.model import load_model_from_path
from ares.dataset import cifar10, imagenet, dataset_to_iterator


class ModelRunner(object):
    def __init__(self, model_path, batch_size, dataset_size, max_forward, max_backward, dataset, output_path):
        self.batch_size, self.dataset_size, self.dataset, self.output_path = batch_size, dataset_size, dataset, output_path
        # load tensorflow session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        # load model
        rs_model = load_model_from_path(model_path)
        self.model = rs_model.load(self.session)
        # prepare tensorflow graph
        self.xs_ph = tf.placeholder(dtype=self.model.x_dtype, shape=(None, *self.model.x_shape))
        self.logits_op, self.labels_op = self.model.logits_and_labels(self.xs_ph)
        self.grad_ys_ph = tf.placeholder(dtype=self.model.x_dtype, shape=(None, self.model.n_class))
        self.grad_op = tf.gradients(ys=self.logits_op, xs=[self.xs_ph], grad_ys=[self.grad_ys_ph],
                                    unconnected_gradients="zero")[0]
        # initialize counters
        self.forward_counter, self.max_forward = 0, max_forward
        self.backward_counter, self.max_backward = 0, max_backward
        # load mpi & sending parameters
        self.comm = MPI.COMM_SELF.Spawn(sys.executable, args=[os.path.join(BASE_PATH, "contest", "loader.py")],
                                        maxprocs=1)
        model_info = (self.model.n_class, self.model.x_min, self.model.x_max, self.model.x_shape,
                      self.model.x_dtype, self.model.y_dtype)
        self.comm.send((model_info, batch_size, dataset_size, max_forward, max_backward, dataset, output_path), dest=0)
        # quota flag
        self.exceed_quota = False

    def quota_not_exceed(self, forward_counter, backward_counter):
        if self.exceed_quota:  # once exceed quota, further operations all return empty
            return False
        else:
            if forward_counter <= self.max_forward and backward_counter <= self.max_backward:
                return True
            else:  # either quota exceed, mark the flag
                self.exceed_quota = True
                return False

    def run(self):
        while True:
            req_type, xs_length = self.comm.recv(source=0)
            if req_type == 0:  # forward
                self.handle_forward(xs_length)
            elif req_type == 1:  # backward
                self.handle_backward(xs_length)
            else:
                err_msg = xs_length
                return err_msg
            # send current counters back
            self.comm.send((self.exceed_quota, self.forward_counter, self.backward_counter), dest=0)

    def evaluate(self):
        # calculate accuracy
        xs_ph = tf.placeholder(runner.model.x_dtype, shape=(None, *runner.model.x_shape))
        labels_op = runner.model.labels(xs_ph)

        # load dataset
        if self.dataset == "cifar10":
            eps = 8.0 / 255.0
            dataset = cifar10.load_dataset_for_classifier(classifier=self.model, offset=0, load_target=False).take(
                self.dataset_size)
        else:  # self.dataset == "imagenet"
            eps = 4.0 / 255.0
            dataset = imagenet.load_dataset_for_classifier(classifier=self.model, offset=0, load_target=False).take(
                self.dataset_size)

        results = np.load(os.path.join(self.output_path, "{:.3f}.npy".format(eps)))
        success_count = 0
        for xs_adv, (_, xs, ys) in zip(results, dataset_to_iterator(dataset.batch(self.batch_size), self.session)):
            assert np.shape(xs_adv) == np.shape(xs)
            xs_adv = xs_adv.astype(np.float32)
            xs_adv = np.clip(xs_adv,
                             xs - eps * (self.model.x_max - self.model.x_min),
                             xs + eps * (self.model.x_max - self.model.x_min))
            xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)
            labels = self.session.run(self.labels_op, feed_dict={self.xs_ph: xs_adv})
            success_count += np.sum(np.logical_not(np.equal(labels, ys)))
        return success_count

    def handle_forward(self, xs_length):
        # need to receive xs anyway
        xs_np = np.empty(shape=(xs_length, *self.model.x_shape), dtype=self.model.x_dtype.as_numpy_dtype)
        self.comm.Recv(xs_np, source=0)
        # check if we would exceed forward quota
        new_forward_counter = self.forward_counter + xs_length
        if self.quota_not_exceed(new_forward_counter, self.backward_counter):
            # forward pass
            self.forward_counter = new_forward_counter
            logits_np = self.session.run(self.logits_op, feed_dict={self.xs_ph: xs_np})
        else:
            # make empty response
            logits_np = np.empty(shape=(xs_length, self.model.n_class), dtype=self.model.x_dtype.as_numpy_dtype)
        # send response
        self.comm.Send(logits_np, dest=0)

    def handle_backward(self, xs_length):
        # need to receive xs anyway
        xs_np = np.empty(shape=(xs_length, *self.model.x_shape), dtype=self.model.x_dtype.as_numpy_dtype)
        self.comm.Recv(xs_np, source=0)
        # need to receive grad_ys anyway
        grad_ys_np = np.empty(shape=(xs_length, self.model.n_class), dtype=self.model.x_dtype.as_numpy_dtype)
        self.comm.Recv(grad_ys_np, source=0)
        # check if we would exceed backward quota
        new_backward_counter = self.backward_counter + xs_length
        if self.quota_not_exceed(self.forward_counter, new_backward_counter):
            # backward pass
            self.backward_counter = new_backward_counter
            grad_np = self.session.run(self.grad_op, feed_dict={self.xs_ph: xs_np, self.grad_ys_ph: grad_ys_np})
        else:
            # make empty response
            grad_np = np.empty(shape=(xs_length, *self.model.x_shape), dtype=self.model.x_dtype.as_numpy_dtype)
        # send response
        self.comm.Send(grad_np, dest=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--result-file', type=str, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--dataset-size', type=int, required=True)
    parser.add_argument('--max-forward', type=int, required=True)
    parser.add_argument('--max-backward', type=int, required=True)
    args = parser.parse_args()
    runner = ModelRunner(args.model_path, args.batch_size, args.dataset_size, args.max_forward, args.max_backward,
                         args.dataset, args.output_path)

    err_msg = None
    total_time = 0
    success_count = 0
    try:
        start_time = time.time()
        err_msg = runner.run()
        if err_msg is None:
            success_count = runner.evaluate()
            end_time = time.time()
            total_time = end_time - start_time
    except Exception as e:
        err_msg = e
    finally:
        path = pathlib.Path(args.output_path)
        if path.exists():
            shutil.rmtree(args.output_path)

    if err_msg is not None:
        info = {"success": "0", "model": args.model_path, "dataset": args.dataset, "total_time": "0",
                "success_count": str(success_count), "err_msg": err_msg}
    else:
        info = {"success": "1", "model": args.model_path, "dataset": args.dataset, "total_time": str(total_time),
                "success_count": str(success_count), "err_msg": err_msg}

    with open(args.result_file, "a+") as f:
        f.write(json.dumps(info) + "\n")
