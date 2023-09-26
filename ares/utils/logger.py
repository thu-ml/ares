import os
import sys
import logging
import ares
from torch.utils._contextlib import _DecoratorContextManager

class CustomFormatter(logging.Formatter):
    """Class for custom formatter."""
    def format(self, record):
        """Directly output message without formattion when got 'simple' attribute."""
        if hasattr(record, 'simple') and record.simple:
            return record.getMessage()
        else:
            return logging.Formatter.format(self, record)


def setup_logger(save_dir=None, distributed_rank=0, main_only=True):
    '''Setup custom logger to record information.'''
    name = ares.__package_name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-main process
    if distributed_rank > 0 and main_only:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = CustomFormatter("[%(asctime)s %(name)s] %(levelname)s: %(message)s", '%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

class format_print(_DecoratorContextManager):
    '''This class is used as a decrator to format output of print func using our custom logger.'''
    def __enter__(self):
        self.pre_stdout = sys.stdout
        sys.stdout = PrintFormatter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.pre_stdout


class PrintFormatter:
    '''This class is used to overwrite the sys.stdout using our custom logger.'''
    def __init__(self):
        self.logger = logging.getLogger(name=ares.__package_name__)

    def write(self, message):
        if message != '\n':
            self.logger.info(message)

    def flush(self):
        pass


