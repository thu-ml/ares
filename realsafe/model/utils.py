import os
import importlib


def get_res(path):
    ''' Get resource's path. '''
    prefix = os.environ.get('REALSAFE_RES_DIR')
    if prefix is None:
        prefix = os.path.expanduser('~/.realsafe/')
    return os.path.abspath(os.path.join(prefix, path))


def load_model_from_path(path):
    ''' TODO '''
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location('rs_model', path)
    rs_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs_model)
    return rs_model
