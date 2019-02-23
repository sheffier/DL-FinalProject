import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # your Project Root


def local_path_to(relative_path):
    return os.path.join(ROOT_DIR, relative_path)


ORG_TRAIN_DATA_PATH = local_path_to('data/original_data/train')
ORG_VALID_DATA_PATH = local_path_to('data/original_data/valid')
ORG_TEST_DATA_PATH = local_path_to('data/original_data/test')
PRC_DATA_PATH = local_path_to('data/processed_data')
PRC_TRAIN_DATA_PATH = local_path_to(PRC_DATA_PATH + '/train')
PRC_VALID_DATA_PATH = local_path_to(PRC_DATA_PATH + '/valid')
PRC_TEST_DATA_PATH = local_path_to(PRC_DATA_PATH + '/test')
