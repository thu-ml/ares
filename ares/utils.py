import os
from tqdm import tqdm
from urllib.request import urlretrieve


def get_res_path(path):
    ''' Get resource's full path. By default, all resources are downloaded into ``~/.ares``. This location could be
    override by the ``ARES_RES_DIR`` environment variable.
    '''
    prefix = os.environ.get('ARES_RES_DIR')
    if prefix is None:
        prefix = os.path.expanduser('~/.ares/')
    return os.path.abspath(os.path.join(prefix, path))


def download_res(url, filename, show_progress_bar=True):
    ''' Download resource at ``url`` and save it to ``path``. If ``show_progress_bar`` is true, a progress bar would be
    displayed.
    '''
    hook = None if not show_progress_bar else _download_res_tqdm_hook(tqdm(unit='B', unit_scale=True))
    urlretrieve(url, filename, hook)


def _download_res_tqdm_hook(pbar):
    ''' Wrapper for tqdm. '''
    downloaded = [0]

    def update(count, block_size, total_size):
        if total_size is not None:
            pbar.total = total_size
        delta = count * block_size - downloaded[0]
        downloaded[0] = count * block_size
        pbar.update(delta)

    return update
