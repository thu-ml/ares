__package_name__ = 'ARES 2.0'
__version__ = '2.0.0'

import os
from .attack import *
from .dataset import *
from .defense import *
from .model import *
from .utils import *
from .utils.registry import registry


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache_dir = os.environ['ARES_CACHE'] if os.environ.get('ARES_CACHE') else os.path.join(root_dir, 'cache')

registry.register_path("root_dir", root_dir)
registry.register_path("cache_dir", cache_dir)
