import re
import os
import torch
import config
import mmap
from bpemb import BPEmb
from contextlib import ExitStack
from typing import Dict, List
from src.data import LabelDict, BpeWordDict, ArticleRawDataset, InfoboxRawDataset,\
                     Article, Infobox, BoxRecord
from collections import defaultdict
from src.utils import safe_mkdir
from tqdm import tqdm
from src.utils import get_num_lines


class PreprocessMetadata(object):
    def __init__(self, emb_dim, word_vocab_size, word_dict_path, field_dict_path):
        self.emb_dim = emb_dim
        self.word_vocab_size = word_vocab_size
        self.word_dict_path = word_dict_path
        self.field_dict_path = field_dict_path

    def init_bpe_module(self):
        return BPEmb(lang="en", dim=self.emb_dim, vs=self.word_vocab_size)


def prepare_articles_dataset(label_dict: LabelDict, bpe: BPEmb, skipped_boxes):
    articles_datasets: Dict[str, ArticleRawDataset] = {'train': None,
                                                       'valid': None,
                                                       'test': None}

    sentences_paths = {'train': {'sentences': config.ORG_TRAIN_DATA_PATH + '/train.sent',
                                 'sents_per_art': config.ORG_TRAIN_DATA_PATH + '/train.nb'},
                       'valid': {'sentences': config.ORG_VALID_DATA_PATH + '/valid.sent',
                                 'sents_per_art': config.ORG_VALID_DATA_PATH + '/valid.nb'},
                       'test': {'sentences': config.ORG_TEST_DATA_PATH + '/test.sent',
                                'sents_per_art': config.ORG_TEST_DATA_PATH + '/test.nb'}}

    article_processed_paths = {'train': config.PRC_TRAIN_DATA_PATH + "/train.article",
                               'valid': config.PRC_VALID_DATA_PATH + "/valid.article",
                               'test': config.PRC_TEST_DATA_PATH + "/test.article"}

    for name in articles_datasets.keys():
        if os.path.isfile(article_processed_paths[name] + '.bin'):
            print("Loading %s Article dataset from %s ..." % (name, article_processed_paths[name] + '.bin'))
            articles_datasets[name] = torch.load(article_processed_paths[name] + '.bin')
            print("Dataset contains %d articles" % len(articles_datasets[name].articles))
        else:
            articles_datasets[name] = ArticleRawDataset(label_dict)

            print("Preprocessing %s articles" % name)

            with ExitStack() as stack:
                f_sents = stack.enter_context(open(sentences_paths[name]['sentences'], "r", encoding='utf-8'))
                f_sents_per_art = stack.enter_context(
                    open(sentences_paths[name]['sents_per_art'], "r", encoding='utf-8'))

                article_cnt = 0

                for idx, line in enumerate(tqdm(f_sents_per_art, total=get_num_lines(sentences_paths[name]['sents_per_art']))):
                    article_cnt += 1

                    if len(skipped_boxes[name]) > 0:
                        if skipped_boxes[name][0] == idx:
                            skipped_boxes[name].pop(0)
                            continue

                    sents_cnt = int(line.strip())
                    article = Article()

                    for cnt in range(sents_cnt):
                        sentence = f_sents.readline().strip()

                        article.add_sentence(sentence)

                    if len(article.sentences) != 0:
                        articles_datasets[name].add_article(article)
                    else:
                        print("article %d was not added" % idx)

            print("[%s] %d / %d articles were added" % (name, len(articles_datasets[name].articles), article_cnt))
            print("Save articles dataset as binary")
            torch.save(articles_datasets[name], article_processed_paths[name] + '.bin')
            print("[%s] Done" % name)

        if os.path.isfile(article_processed_paths[name] + '.content') is False:
            articles_datasets[name].dump(article_processed_paths[name], bpe)
            print("Finished preprocessing. %s dataset has %d articles" % (name, len(articles_datasets[name].articles)))

    del articles_datasets

def prepare_infobox_datasets(label_dict: LabelDict, bpe: BPEmb):
    ib_datasets: Dict[str, InfoboxRawDataset] = {'train': None,
                                                 'valid': None,
                                                 'test': None}

    ib_paths = {'train': config.ORG_TRAIN_DATA_PATH + "/train.box",
                'valid': config.ORG_VALID_DATA_PATH + "/valid.box",
                'test': config.ORG_TEST_DATA_PATH + "/test.box"}

    ib_processed_paths = {'train': config.PRC_TRAIN_DATA_PATH + "/train.box",
                          'valid': config.PRC_VALID_DATA_PATH + "/valid.box",
                          'test': config.PRC_TEST_DATA_PATH + "/test.box"}

    for name in ib_datasets.keys():
        if os.path.isfile(ib_processed_paths[name] + '.bin'):
            print("Loading %s Infobox dataset from %s ..." % (name, ib_processed_paths[name] + '.bin'))
            ib_datasets[name] = torch.load(ib_processed_paths[name] + '.bin')
            print("Dataset contains %d boxes" % len(ib_datasets[name].infoboxes))
        else:
            ib_datasets[name] = InfoboxRawDataset(label_dict)

            print("Preprocessing %s boxes" % name)

            with open(ib_paths[name], "r", encoding='utf-8') as boxes_file:
                infobox_cnt = 0

                for idx, line in enumerate(tqdm(boxes_file, total=get_num_lines(ib_paths[name]))):
                    line = line.strip().split()

                    infobox = Infobox()
                    box_record = None

                    for field_pos_val in line:
                        # Match the following patterns:
                        # label_pos:value
                        # label:value
                        #
                        # * A label in a string of letters and/or "_". Must start with a letter
                        # * Value can be any string as long as it doesn't contains ":".
                        #   Also, the value "<none>" is illegal
                        match = re.search(
                            r"(?i)^([\W_]*)(?P<label>[a-zA-Z_]+[a-zA-Z])(_)?(?(3)(?P<pos>[1-9]\d*))\b:(?P<value>(?!<none>)[^:]+)$",
                            field_pos_val)

                        if match:
                            new_record = match.group('pos') in ['1', None]

                            if new_record:
                                if box_record is not None:
                                    infobox.add_record(box_record)
                                    box_record = None

                                new_label = match.group('label')

                                if -1 == new_label.lower().find('imag'):
                                    box_record = BoxRecord(new_label)

                            if box_record is not None:
                                box_record.add_word(match.group('value'))

                    if box_record is not None:
                        infobox.add_record(box_record)

                    if len(infobox.records) != 0:
                        ib_datasets[name].add_infobox(infobox)
                    else:
                        ib_datasets[name].add_skipped(idx)
                        print("infobox %d was not added" % infobox_cnt)

                    infobox_cnt += 1

            print("[%s] %d / %d boxes were added" % (name, len(ib_datasets[name].infoboxes), infobox_cnt))
            print("[%s] Save box dataset as binary..." % name)
            torch.save(ib_datasets[name], ib_processed_paths[name] + '.bin')
            print("[%s] Done" % name)

        if os.path.isfile(ib_processed_paths[name] + '.content') is False:
            ib_datasets[name].dump(ib_processed_paths[name], bpe)
            print("Finished preprocessing. %s dataset has %d boxes" % (name, len(ib_datasets[name].infoboxes)))

    skipped_boxes = {'train': ib_datasets['train'].skipped_boxes,
                     'valid': ib_datasets['valid'].skipped_boxes,
                     'test': ib_datasets['test'].skipped_boxes}

    del ib_datasets

    return skipped_boxes


def create_field_label_vocab(in_path, out_path):
    if os.path.isfile(out_path):
        print("Loading vocab from %s ..." % out_path)
        vocab = torch.load(out_path)
    else:
        print("Building vocabulary...")
        vocab = defaultdict(int)

        with open(in_path, "r", encoding='utf-8') as box_file:
            for line in tqdm(box_file, total=get_num_lines(in_path)):
                line = line.strip().split()

                for item in line:
                    match = re.search(
                        r"(?i)^([\W_]*)(?P<label>[a-zA-Z_]+[a-zA-Z])(_)?(?(3)(?P<pos>[1-9]\d*))\b:[^:]*$",
                        item)

                    if match:
                        vocab[match.group('label')] += 1

        print("Saving vocabulary to file...")
        torch.save(vocab, out_path)

    print("Field vocab contains %d labels" % len(vocab))
    return vocab


def create_mono_datasets(label_dict: LabelDict, bpe):
    print("Creating Articles mono datasets")
    article_para_ds = {'train': config.PRC_TRAIN_DATA_PATH + "/train.article.bin",
                       'valid': config.PRC_VALID_DATA_PATH + "/valid.article.bin",
                       'test': config.PRC_TEST_DATA_PATH + "/test.article.bin"}

    all_articles = ArticleRawDataset(label_dict)

    for name, dataset_path in article_para_ds.items():
        assert os.path.isfile(dataset_path)

        articles_dataset: ArticleRawDataset = torch.load(dataset_path)
        all_articles.articles.extend(articles_dataset.articles)
        del articles_dataset

    num_entries = len(all_articles.articles)

    train_entries = 2 * ((int(num_entries * 0.8) + 1) // 2)
    valid_entries = (num_entries - train_entries) // 2
    test_entries = num_entries - valid_entries

    start_entry = 0

    mono_art_ds = {'train': [train_entries // 2, train_entries, config.PRC_TRAIN_DATA_PATH + "/train.article.mono"],
                   'valid': [0, valid_entries, config.PRC_VALID_DATA_PATH + "/valid.article.mono"],
                   'test':  [0, test_entries, config.PRC_TEST_DATA_PATH + "/test.article.mono"]}

    for name, article_info in mono_art_ds.items():
        if os.path.isfile(article_info[2] + '.bin') is False:
            article_ds = ArticleRawDataset(label_dict)
            article_ds.articles.extend(all_articles.articles[start_entry + article_info[0]: start_entry + article_info[1]])

            print(name + ": " + "Save mono article dataset as binary")
            article_ds.dump(article_info[2], bpe)
            torch.save(article_ds, article_info[2] + '.bin')
            del article_ds

        start_entry += article_info[1]

    del all_articles

    print("Creating Infobox mono datasets")
    ib_para_ds = {'train': config.PRC_TRAIN_DATA_PATH + "/train.box.bin",
                  'valid': config.PRC_VALID_DATA_PATH + "/valid.box.bin",
                  'test': config.PRC_TEST_DATA_PATH + "/test.box.bin"}

    all_infoboxes = InfoboxRawDataset(label_dict)

    for name, dataset_path in ib_para_ds.items():
        assert os.path.isfile(dataset_path)

        boxes_dataset: InfoboxRawDataset = torch.load(dataset_path)
        all_infoboxes.infoboxes.extend(boxes_dataset.infoboxes)
        del boxes_dataset

    assert num_entries == len(all_infoboxes.infoboxes)

    start_entry = 0

    mono_box_ds = {'train': [0, train_entries // 2, config.PRC_TRAIN_DATA_PATH + "/train.box.mono"],
                   'valid': [0, valid_entries, config.PRC_VALID_DATA_PATH + "/valid.box.mono"],
                   'test':  [0, test_entries, config.PRC_TEST_DATA_PATH + "/test.box.mono"]}

    for name, box_info in mono_box_ds.items():
        if os.path.isfile(box_info[2] + '.bin') is False:
            box_ds = InfoboxRawDataset(label_dict)
            box_ds.infoboxes.extend(all_infoboxes.infoboxes[start_entry + box_info[0]: start_entry + box_info[1]])

            print(name + ": " + "Save mono box dataset as binary")
            box_ds.dump(box_info[2], bpe)
            torch.save(box_ds, box_info[2] + '.bin')
            del box_ds

        start_entry += box_info[1]

    del all_infoboxes


def make_dirs():
    # os.mkdir("../data/results/")
    # os.mkdir("../data/results/res/")
    # os.mkdir("../data/results/evaluation/")

    safe_mkdir(config.PRC_DATA_PATH)
    safe_mkdir(config.PRC_TRAIN_DATA_PATH)
    safe_mkdir(config.PRC_VALID_DATA_PATH)
    safe_mkdir(config.PRC_TEST_DATA_PATH)

    # os.mkdir("../data/processed_data/test/test_split_for_rouge/")
    # os.mkdir("../data/processed_data/valid/valid_split_for_rouge/")


def preprocess(emb_dim, word_vocab_size):
    make_dirs()

    box_file_path = config.ORG_TRAIN_DATA_PATH + "/train.box"
    field_vocab_path = config.PRC_TRAIN_DATA_PATH + "/field.vocab"
    field_dict_path = config.PRC_TRAIN_DATA_PATH + "/field.dict"
    word_dict_path = config.PRC_TRAIN_DATA_PATH + "/word.dict"

    bpemb_en = BPEmb(lang="en", dim=emb_dim, vs=word_vocab_size)

    metadata = PreprocessMetadata(emb_dim, word_vocab_size, word_dict_path, field_dict_path)
    metadata.init_bpe_module()

    field_vocab = create_field_label_vocab(box_file_path, field_vocab_path)
    field_dict = LabelDict.get(vocab=list(field_vocab),
                               dict_binpath=field_dict_path)
    bpe_dict = BpeWordDict.get(vocab=bpemb_en.words, dict_binpath=word_dict_path)

    print("Saving metadata")
    torch.save(metadata, config.PRC_TRAIN_DATA_PATH + '/metadata.bin')

    skipped_boxes = prepare_infobox_datasets(field_dict, bpemb_en)
    prepare_articles_dataset(field_dict, bpemb_en, skipped_boxes)

    create_mono_datasets(field_dict, bpemb_en)

    print("Preprocessing done")

    return bpemb_en, bpe_dict, field_dict


if __name__ == '__main__':
    preprocess(300, 10000)

