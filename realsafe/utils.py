from tqdm import tqdm
from urllib.request import urlretrieve


def download_url(url, filename, show_progress_bar=True):
    '''
    Download file at `url` and save it to `path`. If `show_progress_bar` is true, a progress bar would be displayed.
    '''
    hook = None if not show_progress_bar else _download_url_tqdm_hook(tqdm(unit='B', unit_scale=True))
    urlretrieve(url, filename, hook)


def _download_url_tqdm_hook(pbar):
    ''' Wrapper for tqdm. '''
    downloaded = [0]

    def update(count, block_size, total_size):
        if total_size is not None:
            pbar.total = total_size
        delta = count * block_size - downloaded[0]
        downloaded[0] = count * block_size
        if downloaded[0] < total_size:
            pbar.update(delta)
        else:
            pbar.finish()

    return update
