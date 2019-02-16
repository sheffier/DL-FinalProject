import re, time, os
import torch
from bpemb import BPEmb
from contextlib import ExitStack
from typing import List, Dict, Set
import pandas as pd


FIELD_PAD, FIELD_UNK, FIELD_NULL = '<PAD>', '<UNK>', '<NULL>'
SPECIAL_FIELD_SYMS = 3

BPE_UNK, BPE_BOS, BPE_EOS, WORD_PAD, WORD_SOT = '<unk>', '<s>', '</s>', '<pad>', '<sot>'
SPECIAL_WORD_SYMS = 5


class Dictionary(object):
    def __init__(self, word2id: Dict, id2word: Dict):
        self.word2id = word2id
        self.id2word = id2word

        # self.n_words = len(self.id2word)

    @staticmethod
    def read_vocab(vocab: Set):
        pass

    # def addSentence(self, sentence):
    #     for word in sentence.split(' '):
    #         self.addWord(word)
    #
    # def addWord(self, word):
    #     if word not in self.word2id:
    #         self.word2id[word] = self.n_words
    #         self.word2count[word] = 1
    #         self.id2word[self.n_words] = word
    #         self.n_words += 1
    #     else:
    #         assert word != PAD and word != UNK and word != NULL
    #         self.word2count[word] += 1
    #
    # def saveDict(self, path, cutoff=None):
    #     txt_path = path + '.txt'
    #     bin_path = path + '.bin'
    #
    #     df = pd.DataFrame.from_dict(self.word2count, columns=['freq'], orient='index')
    #     df = df.sort_values('freq', ascending=False)
    #     if cutoff is not None:
    #         df = df.loc[df['freq'] > cutoff]
    #     df.to_csv(txt_path, encoding='utf-8', header=None)
    #
    #     print("Saving the data to %s ..." % bin_path)
    #     torch.save(self, bin_path)


class LabelDict(Dictionary):
    def __init__(self, name: str, word2id: Dict, id2word: Dict):
        super().__init__(word2id, id2word)
        self.name = name
        self.pad_index = self.word2id[FIELD_PAD]
        self.unk_index = self.word2id[FIELD_UNK]
        self.null_index = self.word2id[FIELD_NULL]

    @staticmethod
    def read_vocab(vocab: Set):
        word2id = {FIELD_PAD: 0, FIELD_UNK: 1, FIELD_NULL: 2}

        for idx, label in enumerate(vocab):
            assert label != FIELD_PAD and label != FIELD_UNK and label != FIELD_NULL
            word2id[label] = SPECIAL_FIELD_SYMS + idx

        id2word = {v: k for k, v in word2id.items()}

        return LabelDict("Field", word2id, id2word)


class BpeWordDict(Dictionary):
    def __init__(self, name: str, word2id: Dict, id2word: Dict):
        super().__init__(word2id, id2word)
        self.name = name
        self.unk_index = self.word2id[BPE_UNK]
        self.bos_index = self.word2id[BPE_BOS]
        self.eos_index = self.word2id[BPE_EOS]
        self.pad_index = self.word2id[WORD_PAD]
        self.sot_index = self.word2id[WORD_SOT]

    # Add this to Dictionary class instead
    def __len__(self):
        """Returns the number of words in the dictionary"""
        return len(self.id2word)

    @staticmethod
    def read_vocab(vocab: Set):
        word2id = {}
        for idx, word in enumerate(vocab):
            word2id[word] = idx

        word2id[WORD_PAD] = idx + 1
        word2id[WORD_SOT] = idx + 2
        id2word = {v: k for k, v in word2id.items()}

        return BpeWordDict("Bpe words", word2id, id2word)


class Article(object):
    def __init__(self):
        self.sentences = []

    @staticmethod
    def norm_sentence(sentence):
        return sentence.lower()

    @staticmethod
    def validate_sentence(sentence):
        match = re.search(r"[A-Z]+", sentence)
        assert match is None

    def add_sentence(self, sentence):
        #norm_sent = self.norm_sentence(sentence)
        self.validate_sentence(sentence)
        self.sentences.append(sentence.split())

    def sentence_serialize(self):
        pass


class BoxRecord(object):
    def __init__(self, label: str):
        self.field_label = label
        self.content = []
        # self.positions = []

    @staticmethod
    def validate_word(word: str):
        match = re.search(r"[A-Z]+", word)
        assert match is None

    @staticmethod
    def norm_word(word: str):
        return word.lower()

    def add_word(self, word: str):
        self.validate_word(word)
        self.content.append(word)


class Infobox(object):
    def __init__(self):
        self.records: List[BoxRecord] = []

    def add_record(self, record: BoxRecord):
        self.records.append(record)

    def serialize(self):
        pass


class ArticleRawDataset(object):
    def __init__(self, label_dict: LabelDict):
        self.articles: List[Article] = []
        self.label_dict = label_dict

    def add_article(self, article: Article):
        self.articles.append(article)

    def dump(self, dump_path, bpe: BPEmb, max_sents=1):
        with ExitStack() as stack:
            article_content_file = stack.enter_context(open(dump_path + '.content', "w", encoding='utf-8'))
            article_label_file = stack.enter_context(open(dump_path + '.labels', "w", encoding='utf-8'))

            for article in self.articles:
                sent_cnt = min(max_sents, len(article.sentences))

                flat_sents = [word for sentence in article.sentences[:sent_cnt] for word in sentence]
                sent_bpe_ids = [str(bpe_id) for bpe_id in bpe.encode_ids(" ".join(flat_sents))]
                label_ids = [str(self.label_dict.null_index)] * len(sent_bpe_ids)

                article_content_file.write(" ".join(sent_bpe_ids) + '\n')
                article_label_file.write(" ".join(label_ids) + '\n')


class InfoboxRawDataset(object):
    def __init__(self, label_dict: LabelDict):
        self.infoboxes: List[Infobox] = []
        self.label_dict = label_dict

    def add_infobox(self, infobox: Infobox):
        self.infoboxes.append(infobox)

    def dump(self, dump_path, bpe: BPEmb):
        with ExitStack() as stack:
            boxes_content_file = stack.enter_context(open(dump_path + '.content', "w", encoding='utf-8'))
            boxes_label_file = stack.enter_context(open(dump_path + '.labels', "w", encoding='utf-8'))
            boxes_positions_file = stack.enter_context(open(dump_path + '.pos', "w", encoding='utf-8'))

            for infobox in self.infoboxes:
                box_content = []
                box_labels = []
                box_positions = []
                for record in infobox.records:
                    rec_content = " ".join(record.content)
                    rec_bpe_ids = [str(bpe_id) for bpe_id in bpe.encode_ids(rec_content)]
                    num_tokens = len(rec_bpe_ids)
                    if record.field_label in self.label_dict.word2id:
                        rec_label_id = self.label_dict.word2id[record.field_label]
                    else:
                        print("Unknown field label %s" % record.field_label)
                        rec_label_id = self.label_dict.unk_index

                    label_ids = [str(rec_label_id)] * num_tokens
                    positions = [str(num + 1) for num in range(num_tokens)]

                    box_content.extend(rec_bpe_ids)
                    box_labels.extend(label_ids)
                    box_positions.extend(positions)

                boxes_content_file.write(" ".join(box_content) + '\n')
                boxes_label_file.write(" ".join(box_labels) + '\n')
                boxes_positions_file.write(" ".join(box_positions) + '\n')


def sentence_preprocess(max_sentences_per_article = 1, use_bpe_encoding=True):
    sent_rawdata_paths = [{'sent': '../data/original_data/train/train.sent', 'sent_cnt': '../data/original_data/train/train.nb'},
                          {'sent': '../data/original_data/valid/valid.sent', 'sent_cnt': '../data/original_data/valid/valid.nb'},
                          {'sent': '../data/original_data/test/test.sent', 'sent_cnt': '../data/original_data/test/test.nb'}]

    sent_dataset_paths = [{'sent_val': '../data/processed_data/train/train.sent.val', 'sent_label': '../data/processed_data/train/train.sent.lab'},
                          {'sent_val': '../data/processed_data/valid/valid.sent.val', 'sent_label': '../data/processed_data/valid/valid.sent.lab'},
                          {'sent_val': '../data/processed_data/test/test.sent.val', 'sent_label': '../data/processed_data/test/test.sent.lab'}]

    sentence_label = '<none>'

    tokens_per_article = [{}, {}, {}]

    max_sent_len = 0
    max_idx = 0
    for idx, (raw_data_files, dataset) in enumerate(zip(sent_rawdata_paths, sent_dataset_paths)):
        print("Starting sentence preprocessing\n")

        with ExitStack() as stack:
            fsent = stack.enter_context(open(raw_data_files['sent'], "r", encoding='utf-8'))
            fcnt = stack.enter_context(open(raw_data_files['sent_cnt'], "r", encoding='utf-8'))
            fsent_val = stack.enter_context(open(dataset['sent_val'], "w", encoding='utf-8'))
            fsent_lab = stack.enter_context(open(dataset['sent_label'], "w", encoding='utf-8'))

            bytes_read = 0
            total_bytes = os.path.getsize(raw_data_files['sent_cnt'])
            target_bytes = 0
            article_cnt = 0

            for line in fcnt:
                if bytes_read >= target_bytes:
                    print("progress %.3f" % (100.0 * (bytes_read / total_bytes)))
                    target_bytes += total_bytes // 20

                bytes_read += len(line)

                sent_cnt = int(line.strip())
                serialized_sent = []

                for cnt in range(sent_cnt):
                    sent = fsent.readline()

                    if cnt <= max_sentences_per_article:
                        sent = sent.strip().split()
                        serialized_sent.extend(sent)

                if use_bpe_encoding:
                    #serialized_sent = BpeEncode(" ".join(serialized_sent))
                    pass

                tokens_per_article[idx][article_cnt] = len(serialized_sent)
                article_cnt += 1

                if len(serialized_sent) > max_sent_len:
                    max_sent_len = len(serialized_sent)
                    max_idx = idx

                serialized_lab = [sentence_label] * len(serialized_sent)

                fsent_val.write(" ".join(serialized_sent))
                fsent_val.write('\n')
                fsent_lab.write(" ".join(serialized_lab))
                fsent_lab.write('\n')
        print("Current max sentence length is %d" % max_sent_len)

    df_train = pd.DataFrame.from_dict(tokens_per_article[0], columns=['n_tok'], orient='index')
    df_valid = pd.DataFrame.from_dict(tokens_per_article[1], columns=['n_tok'], orient='index')
    df_test = pd.DataFrame.from_dict(tokens_per_article[2], columns=['n_tok'], orient='index')

    # df = df.sort_values('freq', ascending=False)
    # df_filtered = df.loc[df['freq'] > 10]
    # df_filtered.to_csv(vocab_path, encoding='utf-8', header=None)

    print("Max sentence length is %d (at %d)" % (max_sent_len, max_idx))


def prepare_articles_dataset(label_dict: LabelDict, bpe: BPEmb):
    articles_datasets: Dict[str, ArticleRawDataset] = {'train': None,
                                                       'valid': None,
                                                       'test': None}
    sentences_paths = {'train': {'sentences': '../data/original_data/train/train.sent',
                                  'sents_per_art': '../data/original_data/train/train.nb'},
                       'valid': {'sentences': '../data/original_data/valid/valid.sent',
                                  'sents_per_art': '../data/original_data/valid/valid.nb'},
                       'test': {'sentences': '../data/original_data/test/test.sent',
                                  'sents_per_art': '../data/original_data/test/test.nb'}}

    article_processed_paths = {'train': "../data/processed_data/train/train.article",
                               'valid': "../data/processed_data/valid/valid.article",
                               'test': "../data/processed_data/test/test.article"}

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
                f_sents_per_art = stack.enter_context(open(sentences_paths[name]['sents_per_art'], "r", encoding='utf-8'))

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

        print("Finished preprocessing. %s dataset has %d articles" % (name, len(articles_dataset.articles)))


def prepare_infobox_datasets(label_dict: LabelDict, bpe: BPEmb):
    ib_datasets: Dict[str, InfoboxRawDataset] = {'train': None,
                                                 'valid': None,
                                                 'test': None}

    ib_paths = {'train': "../data/original_data/train/train.box",
                'valid': "../data/original_data/valid/valid.box",
                'test': "../data/original_data/test/test.box"}

    ib_processed_paths = {'train': "../data/processed_data/train/train.box",
                          'valid': "../data/processed_data/valid/valid.box",
                          'test': "../data/processed_data/test/test.box"}

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
                        # * Value can be any string as long as it doesn't contains ":". Also, the value "<none>" is illegal
                        match = re.search(r"(?i)^([\W_]*)(?P<label>[a-zA-Z_]+[a-zA-Z])(_)?(?(3)(?P<pos>[1-9]\d*))\b:(?P<value>(?!<none>)[^:]+)$",
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

        print("Finished preprocessing. %s dataset has %d boxes" % (name, len(boxes_dataset.infoboxes)))

    return ib_datasets


def infobox_preprocess(use_bpe_encoding=True):
    """
    Split infoboxes information to three files:
        - *.box.val is the box content (token)
        - *.box.lab is the field type for each token
        - *.box.pos is the position counted from the begining of a field
    Each line in each file corresponds to a single infobox.
    """

    def append_record(record_content):
        if use_bpe_encoding:
            # from sentencepiece import SentencePieceProcessor
            # spm = SentencePieceProcessor()
            # spm.Load(str("../data/wiki.model"))
            # record_content = BpeEncode(" ".join(record_content))
            pass

        record_content_len = len(record_content)
        record_positions = [str(num) for num in range(1, 1 + record_content_len)]
        infobox_serialized_content.extend(record_content)
        infobox_serialized_labels.extend([record_label] * record_content_len)
        infobox_serialized_positions.extend(record_positions)

    ib_in_paths = ["../data/original_data/train/train.box",
                   "../data/original_data/valid/valid.box",
                   "../data/original_data/test/test.box"]

    ib_out_path_dicts = [{'value':    "../data/processed_data/train/train.box.val",
                          'label':    "../data/processed_data/train/train.box.lab",
                          'position': "../data/processed_data/train/train.box.pos"},
                         {'value':    "../data/processed_data/valid/valid.box.val",
                          'label':    "../data/processed_data/valid/valid.box.lab",
                          'position': "../data/processed_data/valid/valid.box.pos"},
                         {'value':    "../data/processed_data/test/test.box.val",
                          'label':    "../data/processed_data/test/test.box.lab",
                          'position': "../data/processed_data/test/test.box.pos"}]

    for ib_in_path, ib_out_path_dict in zip(ib_in_paths, ib_out_path_dicts):
        print("Starting new preprocessing\n")

        with ExitStack() as stack:
            infile = stack.enter_context(open(ib_in_path, "r", encoding='utf-8'))
            ib_val_file = stack.enter_context(open(ib_out_path_dict['value'], "w", encoding='utf-8'))
            ib_lab_file = stack.enter_context(open(ib_out_path_dict['label'], "w", encoding='utf-8'))
            ib_pos_file = stack.enter_context(open(ib_out_path_dict['position'], "w", encoding='utf-8'))

            bytes_read = 0
            total_bytes = os.path.getsize(ib_in_path)
            target_bytes = 0
            for infobox in infile:
                if bytes_read >= target_bytes:
                    print("progress %.3f" % (100.0 * (bytes_read / total_bytes)))
                    target_bytes += total_bytes // 20

                bytes_read += len(infobox)
                infobox = infobox.strip().split()
                infobox_serialized_content = []
                infobox_serialized_labels = []
                infobox_serialized_positions = []
                record_cnt = 0
                record_content = []

                for record_idx, field_pos_val in enumerate(infobox):
                    # Match the following patterns:
                    # label_pos:value
                    # label:value
                    #
                    # * A label in a string of letters and/or "_". Must start with a letter
                    # * Value can be any string as long as it doesn't contains ":". Also, the value "<none>" is illegal
                    match = re.search(r"(?i)^([\W_]*)(?P<label>[a-zA-Z_]+[a-zA-Z])(_)?(?(3)(?P<pos>[1-9]\d*))\b:(?P<value>(?!<none>)[^:]+)$",
                                      field_pos_val)

                    # if field_pos_val.count(':') > 2:
                    #     continue
                    #
                    # match = re.search(r"(?i)(?P<label>[a-zA-Z_]+[a-zA-Z])(_)?(?(2)(?P<pos>[1-9]\d*)):(?P<value>(?!<none>).+)$",
                    #                   field_pos_val)

                    # if match is None:
                    #     continue

                    if match:
                        new_record = match.group('pos') in ['1', None]

                        if new_record:
                            if len(record_content) != 0:
                                append_record(record_content)

                            record_label = match.group('label')
                            record_content = []

                        record_content.append(match.group('value'))
                        record_cnt += 1

                if len(record_content) != 0:
                    append_record(record_content)

                # assert record_cnt > 0
                if record_cnt > 0:
                    ib_val_file.write(" ".join(infobox_serialized_content) + '\n')
                    ib_lab_file.write(" ".join(infobox_serialized_labels) + '\n')
                    ib_pos_file.write(" ".join(infobox_serialized_positions) + '\n')
                else:
                    print("No record was extracted from current infobox")


def create_field_label_vocab():
    box_file_path = "../data/original_data/train/train.box"
    vocab_path = "../data/original_data/train/train.box" + ".fvocab"

    if os.path.isfile(vocab_path):
        print("Loading vocab from %s ..." % vocab_path)
        vocab = torch.load(vocab_path)
        print("Field vocab contains %d labels" % len(vocab))
        return vocab

    vocab = set()
    bytes_read = 0
    target_bytes = 0
    total_bytes = os.path.getsize(box_file_path)

    with open(box_file_path, "r") as box_file:
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

    print("Field vocab contains %d labels" % len(vocab))
    torch.save(vocab, "../data/original_data/train/train.box.fvocab")

    return vocab


def build_field_label_vocab():
    box_file_path = "../data/original_data/train/train.box"
    vocab_path = "../data/original_data/train/field.vocab"

    field_dict = Dictionary("field")
    bytes_read = 0
    target_bytes = 0
    total_bytes = os.path.getsize(box_file_path)

    with open(box_file_path, "r") as box_file:
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
                    field_dict.addWord(match.group('label'))

    field_dict.saveDict(vocab_path)

    return field_dict


def build_word_vocab():
    pass


def tokens_to_ids():
    pass


def make_dirs():
    # os.mkdir("../data/results/")
    # os.mkdir("../data/results/res/")
    # os.mkdir("../data/results/evaluation/")
    os.mkdir("../data/processed_data/")
    os.mkdir("../data/processed_data/train/")
    os.mkdir("../data/processed_data/test/")
    os.mkdir("../data/processed_data/valid/")
    # os.mkdir("../data/processed_data/test/test_split_for_rouge/")
    # os.mkdir("../data/processed_data/valid/valid_split_for_rouge/")


if __name__ == '__main__':
    make_dirs()
    field_vocab = create_field_label_vocab()
    field_dict = LabelDict.read_vocab(field_vocab)
    torch.save(field_dict, '../data/processed_data/train/field.dict')

    bpemb_en = BPEmb(lang="en")
    bpe_dict = BpeWordDict.read_vocab(bpemb_en.words)
    prepare_infobox_datasets(field_dict, bpemb_en)
    prepare_articles_dataset(field_dict, bpemb_en)

    # infobox_preprocess(use_bpe_encoding=False)
    # sentence_preprocess()
    # build_field_label_vocab()

    # prepare_infobox_datasets()
    # prepare_articles_dataset()

    # check_generated_box()
    print("check done")

