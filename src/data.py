# Copyright (C) 2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import re
import os
import torch
import numpy as np
import collections
from bpemb import BPEmb
from contextlib import ExitStack
from typing import List, Dict
from collections import defaultdict


FIELD_PAD, FIELD_UNK, FIELD_NULL = '<PAD>', '<UNK>', '<NULL>'
SPECIAL_FIELD_SYMS = 3

BPE_UNK, BPE_BOS, BPE_EOS, WORD_PAD, WORD_SOT = '<unk>', '<s>', '</s>', '<pad>', '<sot>'
SPECIAL_WORD_SYMS = 5

bpemb_en = BPEmb(lang="en")


class Dictionary(object):
    def __init__(self, word2id: Dict, id2word: Dict):
        self.word2id = word2id
        self.id2word = id2word

    @staticmethod
    def _vocab2dict(vocab: List):
        pass

    def __len__(self):
        """Returns the number of words in the dictionary"""
        return len(self.id2word)

    @classmethod
    def get(cls, vocab=None, dict_binpath=None):
        if dict_binpath is not None and os.path.isfile(dict_binpath):
            print("Loading dictionary from %s ..." % dict_binpath)
            dictionary = torch.load(dict_binpath)
            print("Done")
        elif vocab is not None:
            assert isinstance(vocab, (defaultdict, list))

            if isinstance(vocab, str):
                assert os.path.isfile(vocab)

                print("Loading vocabulary from %s ..." % vocab)
                vocab = torch.load(vocab)

            assert len(vocab) == len(set(vocab))
            print("Converting vocabulary to dictionary")
            dictionary = cls._vocab2dict(vocab)
            if dict_binpath is not None:
                print("Saving to file..")
                torch.save(dictionary, dict_binpath)
        else:
            raise Exception('Both vocab and dict_binpath are None')

        return dictionary


class LabelDict(Dictionary):
    def __init__(self, name: str, word2id: Dict, id2word: Dict):
        super().__init__(word2id, id2word)
        self.name = name
        self.pad_index = self.word2id[FIELD_PAD]
        self.unk_index = self.word2id[FIELD_UNK]
        self.null_index = self.word2id[FIELD_NULL]

    @staticmethod
    def _vocab2dict(vocab: List):
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

    @staticmethod
    def _vocab2dict(vocab: List):
        word2id = {}
        for idx, word in enumerate(vocab):
            word2id[word] = idx

        word2id[WORD_PAD] = len(word2id)
        word2id[WORD_SOT] = len(word2id)
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
        # norm_sent = self.norm_sentence(sentence)
        self.validate_sentence(sentence)
        self.sentences.append(sentence.split())

    def sentence_serialize(self):
        pass


class BoxRecord(object):
    def __init__(self, label: str):
        self.field_label = label
        self.content = []

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

    def dump(self, dump_path, bpe: BPEmb, max_sents=1, shuffle=0):
        with ExitStack() as stack:
            article_content_file = stack.enter_context(open(dump_path + '.content', "w", encoding='utf-8'))
            article_label_file = stack.enter_context(open(dump_path + '.labels', "w", encoding='utf-8'))

            if shuffle:
                indices = (ind for ind in np.random.RandomState(seed=shuffle).permutation(len(self.articles)))
                print("Write dataset to file (shuffled)")
            else:
                indices = range(len(self.articles))
                print("Write dataset to file (no shuffling)")

            for ind in indices:
                article = self.articles[ind]
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

    def dump(self, dump_path, bpe: BPEmb, shuffle=0):
        with ExitStack() as stack:
            boxes_content_file = stack.enter_context(open(dump_path + '.content', "w", encoding='utf-8'))
            boxes_label_file = stack.enter_context(open(dump_path + '.labels', "w", encoding='utf-8'))
            boxes_positions_file = stack.enter_context(open(dump_path + '.pos', "w", encoding='utf-8'))

            if shuffle:
                indices = (ind for ind in np.random.RandomState(seed=shuffle).permutation(len(self.infoboxes)))
                print("Write dataset to file (shuffled)")
            else:
                indices = range(len(self.infoboxes))
                print("Write dataset to file (no shuffling)")

            for ind in indices:
                infobox = self.infoboxes[ind]
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


class CorpusReader:
    def __init__(self, src_word_file, src_field_file, trg_word_file=None, trg_field_file=None, max_sentence_length=80,
                 cache_size=1000):
        assert (trg_word_file is None and trg_field_file is None) or\
               (trg_word_file is not None and trg_field_file is not None)
        self.src_word_file = src_word_file
        self.src_field_file = src_field_file
        self.trg_word_file = trg_word_file
        self.trg_field_file = trg_field_file
        self.epoch = 1
        self.pending = set()
        self.length2pending = collections.defaultdict(set)
        self.next = 0
        self.cache = []
        self.cache_size = cache_size
        self.max_sentence_length = max_sentence_length
        self.validate = False

    def _fill_cache(self):
        self.next = 0
        self.cache = [self.cache[i] for i in self.pending]
        self.pending = set()
        self.length2pending = collections.defaultdict(set)

        print("Cache occupancy %d/%d. Filling cache..." % (len(self.cache), self.cache_size))
        while len(self.cache) < self.cache_size:
            # try:
            #     line = readline()
            # except StopIteration:
            #     line =''
            src_word = self.src_word_file.readline()
            src_field = self.src_field_file.readline()
            trg_word = self.trg_word_file.readline() if self.trg_word_file is not None else src_word
            trg_field = self.trg_field_file.readline() if self.trg_field_file is not None else src_field

            src_word_ids = [int(id) for id in src_word.strip().split()]
            src_field_ids = [int(id) for id in src_field.strip().split()]
            trg_word_ids = [int(id) for id in trg_word.strip().split()]
            trg_field_ids = [int(id) for id in trg_field.strip().split()]

            # src_length = len(tokenize(src_word))
            # trg_length = len(tokenize(trg_word))
            # assert src_length == len(tokenize(src_field))
            # assert trg_length == len(tokenize(trg_field))

            src_length = len(src_word_ids)
            trg_length = len(trg_word_ids)

            assert src_length == len(src_field_ids)
            assert trg_length == len(trg_field_ids)

            if src_word == '' and trg_word == '':
                self.epoch += 1
                self.src_word_file.seek(0)
                self.src_field_file.seek(0)
                if self.trg_word_file is not None:
                    self.trg_word_file.seek(0)
                    self.trg_field_file.seek(0)
            elif 0 < src_length <= self.max_sentence_length and 0 < trg_length <= self.max_sentence_length:
                self.cache.append(((src_length, trg_length), src_word_ids, trg_word_ids, src_field_ids,
                                   trg_field_ids))

        print("Cache filed")
        for i in range(self.cache_size):
            self.pending.add(i)
            self.length2pending[self.cache[i][0]].add(i)

    def _remove(self, index):
        length = self.cache[index][0]
        self.pending.remove(index)
        self.length2pending[length].remove(index)

    @staticmethod
    def _score_length(src, trg, src_min, src_max, trg_min, trg_max):
        return max(abs(src - src_min),
                   abs(src - src_max),
                   abs(trg - trg_min),
                   abs(trg - trg_max))

    def next_batch(self, size):
        if size > self.cache_size:
            raise ValueError('Cache size smaller than twice the batch size')

        if len(self.pending) < self.cache_size / 2:
            self._fill_cache()

        indices = [self.next]
        length = self.cache[self.next][0]
        target_length = length
        src_min = src_max = length[0]
        trg_min = trg_max = length[1]
        self._remove(self.next)
        while len(indices) < size:
            try:
                index = self.length2pending[target_length].pop()
                self.pending.remove(index)
                indices.append(index)
            except KeyError:
                candidates = [(self._score_length(k[0], k[1], src_min, src_max, trg_min, trg_max), k) for k, v in self.length2pending.items() if len(v) > 0]
                target_length = min(candidates)[1]
                src_min = min(src_min, target_length[0])
                src_max = max(src_max, target_length[0])
                trg_min = min(trg_min, target_length[1])
                trg_max = max(trg_max, target_length[1])

        indices = sorted(indices, key=lambda i: self.cache[i][0], reverse=True)

        for i in range(self.next, self.cache_size):
            if i in self.pending:
                self.next = i
                break

        ret_src_sents = [self.cache[i][1] for i in indices]
        ret_trg_sents = [self.cache[i][2] for i in indices]
        ret_src_sents_field = [self.cache[i][3] for i in indices]
        ret_trg_sents_field = [self.cache[i][4] for i in indices]

        return ret_src_sents, ret_trg_sents, ret_src_sents_field, ret_trg_sents_field
        # return [self.cache[i][1] for i in indices], [self.cache[i][2] for i in indices],\
        #        [self.cache[i][3] for i in indices], [self.cache[i][4] for i in indices],


class BacktranslatorCorpusReader:
    def __init__(self, corpus, translator):
        self.corpus = corpus
        self.translator = translator
        self.epoch = corpus.epoch
        self.validate = False

    def next_batch(self, size):
        src_word, trg_word, src_field, trg_field = self.corpus.next_batch(size)
        src_word, src_field = self.translator.greedy(trg_word, trg_field, train=False)
        if self.corpus.epoch > self.epoch:
            self.validate = True

        self.epoch = self.corpus.epoch
        return src_word, trg_word, src_field, trg_field


def tokenize(sentence):
    return sentence.strip().split()
