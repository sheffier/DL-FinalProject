import os
import errno


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
