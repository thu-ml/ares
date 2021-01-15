# Install ARES

We use ARES as our competition development toolkit. Follow https://github.com/thu-ml/ares#installation to install ARES.

# Download model checkpoints and datasets

Change directory to `contest/`. Run `./download_data.sh` to download model checkpoints and datasets. Many of these model checkpoints and datasets need manual downloading. This script also provides instruction for manual downloading.

# Run example attack

We provide an example 10 steps PGD attack in the `attacker/` folder. The `attacker/` folder is a Python module. You need to implement the `attacker.Attacker` class. The `attacker/` is based on TensorFlow 1.15.4. If you prefer using `numpy`, we also provide the same attack based on `numpy` in the `numpy_attacker/` folder.

We provide an easy-to-use evaluation script `contest/run_attacks.py`:

```bash
$ python3 run_attacks.py --help
usage: run_attacks.py [-h] [--models MODELS] [--attacks ATTACKS]
                      [--output OUTPUT] [--mute-stderr]

optional arguments:
  -h, --help         show this help message and exit
  --models MODELS    comma-separated list of models to run in the format of
                     '<dataset>-<model>', e.g. 'imagenet-free_at', default to
                     all models used by competition stage I, they are
                     cifar10-pgd_at, cifar10-feature_scatter,
                     cifar10-robust_overfitting, cifar10-rst, cifar10-fast_at,
                     cifar10-at_he, cifar10-pre_training, cifar10-mmc,
                     cifar10-free_at, cifar10-awp, cifar10-hydra,
                     cifar10-label_smoothing, imagenet-fast_at, imagenet-
                     free_at
  --attacks ATTACKS  comma-separated list of attack folder/package names to
                     run, default to 'attacker'
  --output OUTPUT    output directory, default to the current directory
  --mute-stderr      mute stderr
```

To run the two example attacks we provided on all models, use the following command:

```bash
$ python3 run_attacks.py --attacks attacker,numpy_attacker --output /tmp
```

This command would write scores to `/tmp/attacker.csv` for `attacker/`, and `/tmp/numpy_attacker.csv` for `numpy_attacker/`. Since the two example attacks are two implementations of the same attack algorithm, they would get exactly same scores. We provide the content of `/tmp/attacker.csv` and `/tmp/numpy_attacker.csv` as following:

```bash
cifar10-pgd_at,0.515
cifar10-feature_scatter,0.248
cifar10-robust_overfitting,0.422
cifar10-rst,0.346
cifar10-fast_at,0.492
cifar10-at_he,0.373
cifar10-pre_training,0.4
cifar10-mmc,0.42
cifar10-free_at,0.525
cifar10-awp,0.349
cifar10-hydra,0.369
cifar10-label_smoothing,0.418
cifar10-pgd_at,0.515
cifar10-feature_scatter,0.248
cifar10-robust_overfitting,0.422
cifar10-rst,0.346
cifar10-fast_at,0.492
cifar10-at_he,0.373
cifar10-pre_training,0.4
cifar10-mmc,0.42
cifar10-free_at,0.525
cifar10-awp,0.349
cifar10-hydra,0.369
cifar10-label_smoothing,0.418
imagenet-fast_at,0.684
imagenet-free_at,0.633
```

The first column is the model. The second column is the attack's score on each model. The final score is the average of scores on all models. If you could reproduce the result, you have setup the development toolkit successfully!

# How to count forward and backward

For the following code:

```python
batch_size = 100
xs_ph = tf.placeholder(model.x_dtype, (batch_size, *model.x_shape))
ys_ph = tf.placeholder(model.y_dtype, (batch_size,))
logits, labels = model.logits_and_labels(xs_ph)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ys_ph, logits=logits)
grads = tf.gradients(loss, xs_ph)[0]
```

1. `session.run((logits, labels), feed_dict={xs_ph: xs, ys_ph: ys})` returns the logits, the predictions. It consumes 100 (forward) model predictions quota.
2. `session.run(grads, feed_dict={xs_ph: xs, ys_ph: ys})` returns the gradient of loss w.r.t model inputs. It consumes 100 (backward) gradient calculation quota and 100 (forward) model predictions quota.
3. `session.run((logits, labels, grads), feed_dict={xs_ph: xs, ys_ph: ys})` returns the logits, the predictions, and the gradient of loss w.r.t model inputs. It still consumes 100 (backward) gradient calculation quota and 100 (forward) model predictions quota since the forward pass and backward pass are in same `session.run()`.
4. If you run multiple `session.run()`, the quota consumption are the summation of each `session.run()`'s quota consumption, even if they are evaluated on the same inputs.

# Tips

Due to limitation of TensorFlow, you CANNOT get gradients of multiple loss functions in one `session.run()`. The following code is invalid:

```python
loss_1 = ... # some loss function
loss_2 = ... # another loss function
grad_1 = tf.gradients(loss_1, xs_ph)[0]
grad_2 = tf.gradients(loss_2, xs_ph)[0]
session.run((grad_1, grad_2), feed_dict={xs_ph: xs, ys_ph: ys})
```

You could first `session.run(grad_1, ...)` and then `session.run(grad_2, ...)` instead.

After you submit your code as a zip file of a Python module, we would extract it into a directory called `attacker/` and then running it. That is to say, your Python module would ALWAYS be renamed to `attacker/` regardless your local module name. Avoid using your local module name in your code! Use relative import if needed. For example, your local module name is `my_attacker`:

```
contest/my_attacker/__init__.py
contest/my_attacker/helper.py
```

Use `from .helper import *` instead of `from my_attacker.helper import *` in the `__init__.py`.

# Submit your attack

Pack your attack into a zip archive using the following commands:

```bash
$ cd your_attacker_directory/
$ zip ../attacker.zip -r *
```

Then the `attacker.zip` is ready for submission.

# How we will run submissions

We HAVE TO apply isolation techniques when running submissions, because:

1. We need to run code submitted by players, so we have to isolate players' code from evaluation code;
2. We don't want players to develop attacks specialized for a certain model instead of a general attacks that applies to all models, so we have to isolate players' code from accessing underlying models' information;

These isolation techniques brings some performance degradation as well as setup complexity. We try hard to keep the `run_attacks.py` behave exactly like how we will run submissions on our servers. Usually `run_attacks.py` is enough, but if you do encounter difference between your local `run_attacks.py` scores and our scores, please let us know. We also provide a guide to setup the same environment of our servers as follow (all commands are runned with the `root` user):

## Server hardware and software information

```
OS: 
    CentOS 7.8
    8 vCPU 32 GiB
    NVIDIA V100 16G GiB
lib:
    python 3.7.6 (Anaconda3-2020.02)
    cuda 10.2
    cudnn 7
    openmpi 3.1.0
```

## Create contest user

The `Attacker` will be runned by user named `contest` and you should create it and prepare its home directory first:

```bash
$ useradd -d /home/contest -m contest
$ sudo -u contest mkdir -p /home/contest/results
```

## Install ARES, download model checkpoints and datasets under `/root`

The ARES development toolkit and model checkpoints are put under `/root` so that user `contest` cannot access them. Please follow the "Install ARES" section and "Download model checkpoints and datasets" section to install ARES at `/root/ares`, and download model checkpoints and datasets to `/root`.

## Copy datasets to `/home/contest`

The `Attacker` needs to access datasets, so we need to copy datasets to `/home/contest`.

```bash
$ python3 -m ares.dataset.imagenet
$ mkdir -p /home/contest/.ares /home/contest/.keras
# copy cifar10 & keras data
$ cp /root/.ares/cifar10/target.npy /home/contest/.ares/cifar10/target.npy
$ cp -R /root/.keras /home/contest/.keras
# copy imagenet data
$ cp /root/.ares/imagenet/target.txt /home/contest/.ares/imagenet/target.txt
$ cp /root/.ares/imagenet/val.txt /home/contest/.ares/imagenet/val.txt
$ cp -R /root/.ares/imagenet/ILSVRC2012_img_val /home/contest/.ares/imagenet/ILSVRC2012_img_val
# chmod so that contest user could access them
$ chmod 755 -R /home/contest/.ares /home/contest/.keras
```

## Copy attacker module

Copy your attacker module to the `/home/contest/attacker` folder. You can use the example attack in the `attacker/` or `numpy_attacker/` folder for test purpose:

```bash
$ cp -r /root/ares/contest/attacker /home/contest/attacker
# or
$ cp -r /root/ares/contest/numpy_attacker /home/contest/attacker
# chmod so that contest user could access them
$ chmod 755 -R /home/contest/attacker
```

## Run the attacker module

We provide the checker evaluation script `contest/checker.py`. After all previous setups, you could finally run the checker:

```bash
$ python3 run_checker.py
```

This script will print scores to stdout as `{"score": 0.0056, "message": "", "status": "0"}` format. It will create a uuid with `uuid.uuid4` as task id, then you can check each model's result in `/var/log/attacker/<task_id>.txt` file. The line format as json is:

```
{
    "success": Int,             # The model attacker success or not 0-False 1-true
    "model": String,            # The model path
    "dataset": String,          # The dataset
    "total_time": String,       # Total time
    "success_count": String,    # Number of data attacked success
    "err_msg": String           # Err_msg return from the attacker
}
```
