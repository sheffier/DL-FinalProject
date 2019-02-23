import re
import os
import torch
import config
from bpemb import BPEmb
from contextlib import ExitStack
from typing import Dict
from src.data import LabelDict, BpeWordDict, ArticleRawDataset, InfoboxRawDataset,\
                     Article, Infobox, BoxRecord, bpemb_en


def prepare_articles_dataset(label_dict: LabelDict, bpe: BPEmb):
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

    for name, articles_dataset in articles_datasets.items():
        if os.path.isfile(article_processed_paths[name] + '.bin'):
            print("Loading %s Article dataset from %s ..." % (name, article_processed_paths[name] + '.bin'))
            articles_dataset = torch.load(article_processed_paths[name] + '.bin')
            print("Dataset contains %d articles" % len(articles_dataset.articles))
        else:
            articles_dataset = ArticleRawDataset(label_dict)

            print("Preprocessing %s articles" % name)

            with ExitStack() as stack:
                f_sents = stack.enter_context(open(sentences_paths[name]['sentences'], "r", encoding='utf-8'))
                f_sents_per_art = stack.enter_context(
                    open(sentences_paths[name]['sents_per_art'], "r", encoding='utf-8'))

                bytes_read = 0
                total_bytes = os.path.getsize(sentences_paths[name]['sents_per_art'])
                target_bytes = 0

                for line in f_sents_per_art:
                    if bytes_read >= target_bytes:
                        print("progress %.3f" % (100.0 * (bytes_read / total_bytes)))
                        target_bytes += total_bytes // 20

                    bytes_read += len(line)
                    sents_cnt = int(line.strip())
                    article = Article()

                    for cnt in range(sents_cnt):
                        sentence = f_sents.readline()

                        article.add_sentence(sentence)

                    articles_dataset.add_article(article)

            print("Save articles dataset as binary")
            torch.save(articles_dataset, article_processed_paths[name] + '.bin')

        if os.path.isfile(article_processed_paths[name] + '.content') is False:
            articles_dataset.dump(article_processed_paths[name], bpe)
        if os.path.isfile(article_processed_paths[name] + '.shuffle.content') is False:
            articles_dataset.dump(article_processed_paths[name] + '.shuffle', bpe, shuffle=1)

        print("Finished preprocessing. %s dataset has %d articles" % (name, len(articles_dataset.articles)))


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

    for name, boxes_dataset in ib_datasets.items():
        if os.path.isfile(ib_processed_paths[name] + '.bin'):
            print("Loading %s Infobox dataset from %s ..." % (name, ib_processed_paths[name] + '.bin'))
            boxes_dataset = torch.load(ib_processed_paths[name] + '.bin')
            print("Dataset contains %d boxes" % len(boxes_dataset.infoboxes))
        else:
            boxes_dataset = InfoboxRawDataset(label_dict)

            print("Preprocessing %s boxes" % name)

            with open(ib_paths[name], "r", encoding='utf-8') as boxes_file:
                bytes_read = 0
                total_bytes = os.path.getsize(ib_paths[name])
                target_bytes = 0

                for line in boxes_file:
                    if bytes_read >= target_bytes:
                        print("progress %.3f" % (100.0 * (bytes_read / total_bytes)))
                        target_bytes += total_bytes // 20

                    bytes_read += len(line)
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

                                box_record = BoxRecord(match.group('label'))

                            if box_record is not None:
                                box_record.add_word(match.group('value'))

                    if len(box_record.content) != 0:
                        infobox.add_record(box_record)

                    boxes_dataset.add_infobox(infobox)

            print("Save box dataset as binary")
            torch.save(boxes_dataset, ib_processed_paths[name] + '.bin')

        if os.path.isfile(ib_processed_paths[name] + '.content') is False:
            boxes_dataset.dump(ib_processed_paths[name], bpe)
        if os.path.isfile(ib_processed_paths[name] + '.shuffle.content') is False:
            boxes_dataset.dump(ib_processed_paths[name] + '.shuffle', bpe, shuffle=2)

        print("Finished preprocessing. %s dataset has %d boxes" % (name, len(boxes_dataset.infoboxes)))

    return ib_datasets


def create_field_label_vocab(in_path, out_path):
    if os.path.isfile(out_path):
        print("Loading vocab from %s ..." % out_path)
        vocab = torch.load(out_path)
    else:
        print("Building vocabulary...")
        vocab = set()
        bytes_read = 0
        target_bytes = 0
        total_bytes = os.path.getsize(in_path)

        with open(in_path, "r") as box_file:
            for idx, line in enumerate(box_file):
                if bytes_read >= target_bytes:
                    print("progress %.3f" % (100.0 * (bytes_read / total_bytes)))
                    target_bytes += total_bytes // 20

                bytes_read += len(line)

                line = line.strip().split()

                for item in line:
                    match = re.search(
                        r"(?i)^([\W_]*)(?P<label>[a-zA-Z_]+[a-zA-Z])(_)?(?(3)(?P<pos>[1-9]\d*))\b:[^:]*$",
                        item)

                    if match:
                        vocab.add(match.group('label'))

        print("Saving vocabulary to file...")
        torch.save(vocab, out_path)

    print("Field vocab contains %d labels" % len(vocab))
    return vocab


def make_dirs():
    def safe_mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    # os.mkdir("../data/results/")
    # os.mkdir("../data/results/res/")
    # os.mkdir("../data/results/evaluation/")

    safe_mkdir(config.PRC_DATA_PATH)
    safe_mkdir(config.PRC_TRAIN_DATA_PATH)
    safe_mkdir(config.PRC_VALID_DATA_PATH)
    safe_mkdir(config.PRC_TEST_DATA_PATH)

    # os.mkdir("../data/processed_data/test/test_split_for_rouge/")
    # os.mkdir("../data/processed_data/valid/valid_split_for_rouge/")


if __name__ == '__main__':
    make_dirs()

    box_file_path = config.ORG_TRAIN_DATA_PATH + "/train.box"
    field_vocab_path = config.PRC_TRAIN_DATA_PATH + "/field.vocab"
    field_dict_path = config.PRC_TRAIN_DATA_PATH + "/field.dict"
    word_dict_path = config.PRC_TRAIN_DATA_PATH + "/word.dict"

    field_vocab = create_field_label_vocab(box_file_path, field_vocab_path)
    field_dict = LabelDict.get(vocab=field_vocab,
                               dict_binpath=field_dict_path)

    # bpemb_en = BPEmb(lang="en")
    bpe_dict = BpeWordDict.get(vocab=set(bpemb_en.words), dict_binpath=word_dict_path)

    prepare_infobox_datasets(field_dict, bpemb_en)
    prepare_articles_dataset(field_dict, bpemb_en)

    print("Preprocessing done")
