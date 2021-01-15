import os
import shutil
from urllib.parse import urlparse
import logging
logger = logging.getLogger("checker")

import subprocess32


def file_is_exist(p):
    return os.path.exists(p)


def remove_file(file_path):
    """remove file"""
    if os.path.exists(file_path) and os.path.isfile(file_path):  # 如果文件存在
        os.remove(file_path)
        logger.info("delete file {}".format(file_path))
        return True
    else:
        return False


def remove_dir(dir_path):
    """remove dir"""
    if os.path.exists(dir_path) and os.path.isdir(dir_path):  # 如果文件夹存在
        shutil.rmtree(dir_path)
        logger.info("delete file {}".format(dir_path))
        return True
    else:
        return False


def clear_all_files_in_dir(dir_path):
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        return False

    for i in os.listdir(dir_path):
        path_name = os.path.join(dir_path, i)
        if os.path.isfile(path_name):
            remove_file(path_name)
        else:
            remove_dir(path_name)
    logger.info("clear all files in dir {}".format(dir_path))
    return True


def unzip(source_zip, target_dir):
    try:
        clear_all_files_in_dir(target_dir)
        cmd = ['unzip', '-o', source_zip, '-d', target_dir]
        subprocess32.check_call(cmd)
        logger.info("unzip {} to {}: success".format(source_zip, target_dir))
    except subprocess32.CalledProcessError as e:
        logger.error("unzip {} to {}: error -- {}".format(source_zip, target_dir, e))
        raise Exception('unzip {} failed'.format(source_zip))
    return source_zip


def parse_url_name(url):
    result = urlparse(url)
    p = result.path
    if not p.endswith('.zip'):
        raise ValueError("url resource not a zip file")
    return p.split("/")[-1]


def download(oss_path, local_path):
    name = parse_url_name(oss_path)
    try:
        cmd = ['wget', oss_path, '-O', local_path]
        subprocess32.check_call(cmd)
        logger.info("download {} to {}: success".format(oss_path, local_path))
    except subprocess32.CalledProcessError as e:
        logger.error("download {} to {}: filed -- {}".format(oss_path, local_path, e))
        raise Exception('Download {} failed'.format(oss_path))
    return name
