''' Loader for loading model from a python file. '''

import os
import sys
import importlib


def load_model_from_path(path):
    '''
    Load a python file at ``path`` as a model. A function ``load(session)`` should be defined inside the python file,
    which load the model into the ``session`` and returns the model instance.
    '''
    path = os.path.abspath(path)

    # to support relative import, we add the directory of the target file to path.
    path_dir = os.path.dirname(path)
    if path_dir not in sys.path:
        need_remove = True
        sys.path.append(path_dir)
    else:
        need_remove = False

    spec = importlib.util.spec_from_file_location('rs_model', path)
    rs_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs_model)

    if need_remove:
        sys.path.remove(path_dir)

    return rs_model
