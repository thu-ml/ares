import torch
import shutil
import os
import yaml
import sys
import shutil, errno
from distutils.dir_util import copy_tree
from utils.model import subnet_to_dense


def save_checkpoint(
    state, is_best, args, result_dir, filename="checkpoint.pth.tar", save_dense=False
):
    torch.save(state, os.path.join(result_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(result_dir, filename),
            os.path.join(result_dir, "model_best.pth.tar"),
        )

    if save_dense:
        state["state_dict"] = subnet_to_dense(state["state_dict"], args.k)
        torch.save(
            subnet_to_dense(state, args.k),
            os.path.join(result_dir, "checkpoint_dense.pth.tar"),
        )
        if is_best:
            shutil.copyfile(
                os.path.join(result_dir, "checkpoint_dense.pth.tar"),
                os.path.join(result_dir, "model_best_dense.pth.tar"),
            )


def create_subdirs(sub_dir):
    os.mkdir(sub_dir)
    os.mkdir(os.path.join(sub_dir, "checkpoint"))


def write_to_file(file, data, option):
    with open(file, option) as f:
        f.write(data)


def clone_results_to_latest_subdir(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    copy_tree(src, dst)


# ref:https://github.com/allenai/hidden-networks/blob/master/configs/parser.py
def trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1

    return st[i:]


def arg_to_varname(st: str):
    st = trim_preceding_hyphens(st)
    st = st.replace("-", "_")

    return st.split("=")[0]


def argv_to_vars(argv):
    var_names = []
    for arg in argv:
        if arg.startswith("-") and arg_to_varname(arg) != "config":
            var_names.append(arg_to_varname(arg))

    return var_names


# ref: https://github.com/allenai/hidden-networks/blob/master/args.py
def parse_configs_file(args):
    # get commands from command line
    override_args = argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.configs).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.configs}")
    args.__dict__.update(loaded_yaml)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def write_to_tensorboard(self, writer, prefix, global_step):
        for meter in self.meters:
            writer.add_scalar(f"{prefix}/{meter.name}", meter.val, global_step)

