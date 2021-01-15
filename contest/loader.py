#!/usr/bin/env python3
import os
import sys
BASE_PATH = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
from tensorflow.python.framework.errors_impl import OpError
sys.path.append(BASE_PATH)
os.environ["HOME"] = "/home/contest"
import pwd, grp, sys
from mpi4py import MPI
import traceback
import numpy as np
import tensorflow as tf
from ares.model import ClassifierWithLogits
from ares.dataset import cifar10, imagenet, dataset_to_iterator

# don't take all GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

comm = MPI.Comm.Get_parent()
rank = comm.Get_rank()
assert rank == 0

# receive parameters
model_info, batch_size, dataset_size, max_forward, max_backward, dataset_name, output_path = comm.recv(source=0)


class ProxyClassifierWithLogits(ClassifierWithLogits):
    def __init__(self):
        super().__init__(*model_info)
        self.forward_counter, self.max_forward = 0, max_forward
        self.backward_counter, self.max_backward = 0, max_backward
        self.exceed_quota = False

        @tf.custom_gradient
        def eager_tf_logits(xs_eager):
            # forward pass
            xs_np = xs_eager.numpy()
            xs_length = len(xs_np)
            comm.send((0, xs_length), dest=0)
            comm.Send(xs_np, dest=0)
            logits_np = np.empty(shape=(xs_length, self.n_class), dtype=self.x_dtype.as_numpy_dtype)
            comm.Recv(logits_np, source=0)
            self.exceed_quota, self.forward_counter, self.backward_counter = comm.recv(source=0)

            def eager_tf_logits_grad(d_output_eager):
                # backward pass
                d_output_np = d_output_eager.numpy()
                assert np.shape(d_output_np) == (xs_length, self.n_class)
                comm.send((1, xs_length), dest=0)
                comm.Send(xs_np, dest=0)
                comm.Send(d_output_np, dest=0)
                grad_np = np.empty(shape=(xs_length, *self.x_shape), dtype=self.x_dtype.as_numpy_dtype)
                comm.Recv(grad_np, source=0)
                self.exceed_quota, self.forward_counter, self.backward_counter = comm.recv(source=0)
                return tf.convert_to_tensor(grad_np)

            return tf.convert_to_tensor(logits_np), eager_tf_logits_grad

        self._eager_tf_logits = eager_tf_logits

    def _logits_and_labels(self, xs):
        # wrap the eager mode function into a normal tensorflow op with tf.py_function
        logits = tf.py_function(func=self._eager_tf_logits, inp=[xs], Tout=self.x_dtype)
        logits.set_shape((xs.shape[0], self.n_class))
        labels = tf.argmax(logits, 1, output_type=self.y_dtype)
        return logits, labels


model = ProxyClassifierWithLogits()

# drop priviledge
running_uid = pwd.getpwnam("contest").pw_uid
running_gid = grp.getgrnam("contest").gr_gid
os.setgroups([])
os.setgid(running_gid)
os.setuid(running_uid)
old_umask = os.umask(0o22)

# load dataset
if dataset_name == "cifar10":
    eps = 8.0 / 255.0
    dataset = cifar10.load_dataset_for_classifier(classifier=model, offset=0, load_target=False).take(dataset_size)
else:  # dataset_name == "imagenet"
    eps = 4.0 / 255.0
    dataset = imagenet.load_dataset_for_classifier(classifier=model, offset=0, load_target=False).take(dataset_size)

# create directory for output
os.makedirs(output_path, exist_ok=True)

# run attack with model & session
# assume the attacker module is in /home/contest/attacker/
sys.path.insert(0, "/home/contest")
err_msg = None
try:
    from attacker import Attacker
    
    attacker = Attacker(model, batch_size, dataset_name, session)
    
    attacker.config(magnitude=eps * (model.x_max - model.x_min))
    results = []
    for batch, (_, xs, ys) in enumerate(dataset_to_iterator(dataset.batch(batch_size), session)):
        print("batch={}".format(batch))
        xs_adv = attacker.batch_attack(xs, ys=ys)
        # check if we have exceed quota
        if model.exceed_quota: raise Exception("Exceed quota")
        # to avoid numerical precision causing difference, cast adversarial examples to float32
        xs_adv = xs_adv.astype(np.float32)
        assert np.shape(xs_adv) == np.shape(xs)
        results.append(xs_adv)
    np.save(os.path.join(output_path, "{:.3f}.npy".format(eps)), results)
except OpError as e:
    err_msg = str(type(e))
except Exception as e:
    err_msg = str(e)

comm.send((-1, err_msg), dest=0)

comm.Disconnect()
