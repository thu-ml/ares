import os
import importlib


def get_res_path(path):
    '''
    Get resource's full path. By default, all resources are downloaded into `~/.realsafe`. This location could be
    overrided by the `REALSAFE_RES_DIR` environment variable.
    '''
    prefix = os.environ.get('REALSAFE_RES_DIR')
    if prefix is None:
        prefix = os.path.expanduser('~/.realsafe/')
    return os.path.abspath(os.path.join(prefix, path))


def load_model_from_path(path):
    '''
    Load a python file at `path` as a model. A function `load(session)` should be defined insided the python file, which
    load the model into the `session` and returns the model instance.
    '''
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location('rs_model', path)
    rs_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs_model)
    return rs_model
