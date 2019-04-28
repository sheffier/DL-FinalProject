import os
import errno
import mmap
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project Root


def local_path_to(relative_path):
    return os.path.join(ROOT_DIR, relative_path)


def safe_mkdir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def get_num_lines(file_path):
    with open(file_path, "r+") as fp:
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1

    return lines
