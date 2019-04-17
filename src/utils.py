import os
import errno


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # your Project Root


def local_path_to(relative_path):
    return os.path.join(ROOT_DIR, relative_path)


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
