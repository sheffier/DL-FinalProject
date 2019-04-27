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
import itertools
from bpemb import BPEmb
from contextlib import ExitStack
from typing import List, Dict
from collections import defaultdict


FIELD_PAD, FIELD_UNK, FIELD_NULL = '<PAD>', '<UNK>', '<NULL>'
SPECIAL_FIELD_SYMS = 3

BPE_UNK, BPE_BOS, BPE_EOS, WORD_PAD, WORD_SOT = '<unk>', '<s>', '</s>', '<pad>', '<sot>'
SPECIAL_WORD_SYMS = 5

#bpemb_en = BPEmb(lang="en")
# bpemb_en = BPEmb(lang="en", vs=200000, dim=300)


class Dictionary(object):
    def __init__(self, word2id: Dict, id2word: Dict, pad='<pad>', eos='</s>', unk='<unk>'):
        self.word2id = word2id
        self.id2word = id2word
        self.unk_word = unk

    @staticmethod
    def _vocab2dict(vocab: List):
        pass

    def __len__(self):
        """Returns the number of words in the dictionary"""
        return len(self.id2word)

    def __getitem__(self, idx):
        if idx < len(self.id2word):
            return self.id2word[idx]
        return self.unk_word

    @classmethod
    def get(cls, vocab=None, dict_binpath=None):
        if dict_binpath is not None and os.path.isfile(dict_binpath):
            print("Loading dictionary from %s ..." % dict_binpath)
            dictionary = torch.load(dict_binpath)
            print("Done")
        elif vocab is not None:
            assert isinstance(vocab, (str, list))

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

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index


class LabelDict(Dictionary):
    def __init__(self, name: str, word2id: Dict, id2word: Dict):
        super().__init__(word2id, id2word, pad=FIELD_PAD, unk=FIELD_UNK)
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

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def null(self):
        """Helper to get index of null symbol"""
        return self.null_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index


class BpeWordDict(Dictionary):
    def __init__(self, name: str, word2id: Dict, id2word: Dict):
        super().__init__(word2id, id2word, pad=WORD_PAD, eos=BPE_EOS, unk=BPE_UNK)
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

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def sos(self, src_type):
        """Helper to get index of start-of-sentence symbol"""
        return self.bos_index if src_type is 'text' else self.sot_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index


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

                flat_sents = list(itertools.chain.from_iterable(article.sentences[:sent_cnt]))
                sent_bpe_ids = [str(bpe_id) for bpe_id in bpe.encode_ids(" ".join(flat_sents))]
                label_ids = [str(self.label_dict.null_index)] * len(sent_bpe_ids)

                article_content_file.write(" ".join(sent_bpe_ids) + '\n')
                article_label_file.write(" ".join(label_ids) + '\n')


class InfoboxRawDataset(object):
    def __init__(self, label_dict: LabelDict):
        self.infoboxes: List[Infobox] = []
        self.skipped_boxes: List[int] = []
        self.label_dict = label_dict

    def add_infobox(self, infobox: Infobox):
        self.infoboxes.append(infobox)

    def add_skipped(self, box_index):
        self.skipped_boxes.append(box_index)

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
        self.pending = set()
        self.length2pending = collections.defaultdict(set)
        self.next = 0
        self.cache = []
        self.cache_size = cache_size
        self.max_sentence_length = max_sentence_length

    def _fill_cache(self):
        self.next = 0
        self.cache = [self.cache[i] for i in self.pending]
        self.pending = set()
        self.length2pending = collections.defaultdict(set)

        print("Cache occupancy %d/%d. Filling cache..." % (len(self.cache), self.cache_size))
        while len(self.cache) < self.cache_size:
            src_word = self.src_word_file.readline()
            src_field = self.src_field_file.readline()
            trg_word = self.trg_word_file.readline() if self.trg_word_file is not None else src_word
            trg_field = self.trg_field_file.readline() if self.trg_field_file is not None else src_field

            src_word_ids = [int(id) for id in src_word.strip().split()]
            src_field_ids = [int(id) for id in src_field.strip().split()]
            trg_word_ids = [int(id) for id in trg_word.strip().split()]
            trg_field_ids = [int(id) for id in trg_field.strip().split()]

            src_length = len(src_word_ids)
            trg_length = len(trg_word_ids)

            assert src_length == len(src_field_ids)
            assert trg_length == len(trg_field_ids)

            if src_word == '' and trg_word == '':
                self.src_word_file.seek(0)
                self.src_field_file.seek(0)
                if self.trg_word_file is not None:
                    self.trg_word_file.seek(0)
                    self.trg_field_file.seek(0)
            elif 0 < src_length <= self.max_sentence_length and 0 < trg_length <= self.max_sentence_length:
                self.cache.append(((src_length, trg_length), src_word_ids, trg_word_ids, src_field_ids,
                                   trg_field_ids))

        print("Cache filled")
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
                candidates = [(self._score_length(k[0], k[1], src_min, src_max, trg_min, trg_max), k)
                              for k, v in self.length2pending.items() if len(v) > 0]
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


class BacktranslatorCorpusReader:
    def __init__(self, corpus, translator, beam_size=0):
        self.corpus = corpus
        self.translator = translator
        self.beam_size = beam_size

    def next_batch(self, size):
        src_word, trg_word, src_field, trg_field = self.corpus.next_batch(size)
        if self.beam_size == 0:
            src_word, src_field, _ = self.translator.greedy(trg_word, trg_field, train=False)
        else:
            src_word, src_field = self.translator.beam_search(trg_word, trg_field, beam_size=self.beam_size,
                                                              train=False)

        return src_word, trg_word, src_field, trg_field


class WordNoising(object):
    """Generate a noisy version of a sentence, without changing words themselves."""
    def __init__(self, w_dict, f_dict, bpe_start_marker="▁"):
        self.w_dict = w_dict
        self.f_dict = f_dict
        self.bpe_start = None
        if bpe_start_marker:
            self.bpe_start = np.array([
                self.w_dict[i].startswith(bpe_start_marker)
                for i in range(len(self.w_dict))
            ])

        self.get_word_idx = (
            self._get_bpe_word_idx
            if self.bpe_start is not None
            else self._get_token_idx
        )

    def noising(self, sents, sents_fields, lengths, noising_prob=0.0):
        raise NotImplementedError()

    def _get_bpe_word_idx(self, sents):
        """
        Given a list of BPE tokens, for every index in the tokens list,
        return the index of the word grouping that it belongs to.
        For example, for input sents corresponding to ["▁how", "▁are", "▁y", "ou"],
        return [[0], [1], [2], [2]].
        """
        # sents: (T x B)
        bpe_start = self.bpe_start[sents]

        if (sents.size(0) == 1 and sents.size(1) == 1):
            # Special case when we only have one word in sents. If sents = [[N]],
            # bpe_start is a scalar (bool) instead of a 2-dim array of bools,
            # which makes the sum operation below fail.
            return np.array([[0]])

        # do a reduce front sum to generate word ids
        word_idx = bpe_start.cumsum(0)

        return word_idx

    def _get_token_idx(self, x):  # fix
        """
        This is to extend noising functions to be able to apply to non-bpe
        tokens, e.g. word or characters.
        """
        x = torch.t(x)
        word_idx = np.array([range(len(x_i)) for x_i in x])
        return np.transpose(word_idx)


class WordDropout(WordNoising):
    """Randomly drop input words. If not passing blank_idx (default is None),
    then dropped words will be removed. Otherwise, it will be replaced by the
    blank_idx."""

    def __init__(self, w_dict, f_dict, bpe_start_marker="▁"):
        super().__init__(w_dict, f_dict, bpe_start_marker)

    def noising(self, sents, sents_fields, lengths, dropout_prob=0.1, w_blank_idx=None, f_blank_idx=None):
        # sents: (T x B), lengths: B
        if dropout_prob == 0:
            return sents, sents_fields, lengths

        assert 0 < dropout_prob < 1

        # be sure to drop entire words
        word_idx = self.get_word_idx(sents)
        sentences = []
        sentences_fields = []
        modified_lengths = []
        for i in range(lengths.size(0)):
            # Since dropout probabilities need to apply over non-pad tokens,
            # it is not trivial to generate the keep mask without consider
            # input lengths; otherwise, this could be done outside the loop

            # We want to drop whole words based on word_idx grouping
            num_words = max(word_idx[:, i]) + 1

            # ith example: [x0, x1, ..., eos, pad, ..., pad]
            # We should only generate keep probs for non-EOS tokens. Thus if the
            # input sentence ends in EOS, the last word idx is not included in
            # the dropout mask generation and we append True to always keep EOS.
            # Otherwise, just generate the dropout mask for all word idx
            # positions.
            has_eos = sents[lengths[i] - 1, i] == self.w_dict.eos()
            if has_eos:  # has eos?
                keep = np.random.rand(num_words - 1) >= dropout_prob
                keep = np.append(keep, [True])  # keep EOS symbol
            else:
                keep = np.random.rand(num_words) >= dropout_prob

            words = sents[:lengths[i], i].tolist()
            fields = sents_fields[:lengths[i], i].tolist()

            # TODO: speed up the following loop
            # drop words from the input according to keep
            new_s = [
                w if keep[word_idx[j, i]] else w_blank_idx
                for j, w in enumerate(words)
            ]
            new_f = [
                f if keep[word_idx[j, i]] else f_blank_idx
                for j, f in enumerate(fields)
            ]
            new_s = [w for w in new_s if w is not None]
            new_f = [f for f in new_f if f is not None]
            # we need to have at least one word in the sentence (more than the
            # start / end sentence symbols)
            if len(new_s) <= 1:
                # insert at beginning in case the only token left is EOS
                # EOS should be at end of list.
                rand_idx = np.random.randint(0, len(words))
                new_s.insert(0, words[rand_idx])
                new_f.insert(0, fields[rand_idx])
            assert len(new_s) >= 1 and (
                not has_eos  # Either don't have EOS at end or last token is EOS
                or (len(new_s) >= 2 and new_s[-1] == self.w_dict.eos())
            ), "New sentence is invalid."
            sentences.append(new_s)
            sentences_fields.append(new_f)
            modified_lengths.append(len(new_s))
        # re-construct input
        modified_lengths = torch.LongTensor(modified_lengths)
        modified_sents = torch.LongTensor(
            modified_lengths.max(),
            modified_lengths.size(0)
        ).fill_(self.w_dict.pad())
        modified_fields = torch.LongTensor(
            modified_lengths.max(),
            modified_lengths.size(0)
        ).fill_(self.f_dict.pad())
        for i in range(modified_lengths.size(0)):
            modified_sents[:modified_lengths[i], i].copy_(torch.LongTensor(sentences[i]))
            modified_fields[:modified_lengths[i], i].copy_(torch.LongTensor(sentences_fields[i]))

        return modified_sents, modified_fields, modified_lengths


class WordShuffle(WordNoising):
    """Shuffle words by no more than k positions."""

    def __init__(self, w_dict, f_dict, bpe_start_marker="▁"):
        super().__init__(w_dict, f_dict, bpe_start_marker)

    def noising(self, sents, sents_fields, lengths, max_shuffle_distance=3):
        # sents: (T x B), lengths: B
        if max_shuffle_distance == 0:
            return sents, sents_fields, lengths

        # max_shuffle_distance < 1 will return the same sequence
        assert max_shuffle_distance > 1

        # define noise word scores
        noise = np.random.uniform(
            0,
            max_shuffle_distance,
            size=(sents.size(0), sents.size(1)),
        )
        noise[0] = -1  # do not move start sentence symbol
        # be sure to shuffle entire words
        word_idx = self.get_word_idx(sents)
        sents2 = sents.clone()
        sents_fields2 = sents_fields.clone()
        for i in range(lengths.size(0)):
            length_no_eos = lengths[i]
            if sents[lengths[i] - 1, i] == self.w_dict.eos():
                length_no_eos = lengths[i] - 1
            # generate a random permutation
            scores = word_idx[:length_no_eos, i] + noise[word_idx[:length_no_eos, i], i]
            # ensure no reordering inside a word
            scores += 1e-6 * np.arange(length_no_eos)
            permutation = scores.argsort()
            # shuffle words
            sents2[:length_no_eos, i].copy_(
                sents2[:length_no_eos, i][torch.from_numpy(permutation)]
            )
            sents_fields2[:length_no_eos, i].copy_(
                sents_fields2[:length_no_eos, i][torch.from_numpy(permutation)]
            )
        return sents2, sents_fields2, lengths


class UnsupervisedMTNoising(WordNoising):
    """
    Implements the default configuration for noising in UnsupervisedMT
    (github.com/facebookresearch/UnsupervisedMT)
    """
    def __init__(
        self,
        w_dict,
        f_dict,
        max_word_shuffle_distance,
        word_dropout_prob,
        word_blanking_prob,
        bpe_start_marker="▁"
    ):
        super().__init__(w_dict, f_dict)
        self.max_word_shuffle_distance = max_word_shuffle_distance
        self.word_dropout_prob = word_dropout_prob
        self.word_blanking_prob = word_blanking_prob

        self.word_dropout = WordDropout(
            w_dict=w_dict,
            f_dict=f_dict,
            bpe_start_marker=bpe_start_marker,
        )
        self.word_shuffle = WordShuffle(
            w_dict=w_dict,
            f_dict=f_dict,
            bpe_start_marker=bpe_start_marker,
        )

    def noising(self, sents, sents_fields, lengths):
        # 1. Word Shuffle
        noisy_src_tokens, noisy_src_fields, noisy_src_lengths = self.word_shuffle.noising(
            sents=sents,
            sents_fields=sents_fields,
            lengths=lengths,
            max_shuffle_distance=self.max_word_shuffle_distance,
        )
        # 2. Word Dropout
        noisy_src_tokens, noisy_src_fields, noisy_src_lengths = self.word_dropout.noising(
            sents=noisy_src_tokens,
            sents_fields=noisy_src_fields,
            lengths=noisy_src_lengths,
            dropout_prob=self.word_dropout_prob,
        )
        # 3. Word Blanking
        noisy_src_tokens, noisy_src_fields, noisy_src_lengths = self.word_dropout.noising(
            sents=noisy_src_tokens,
            sents_fields=noisy_src_fields,
            lengths=noisy_src_lengths,
            dropout_prob=self.word_blanking_prob,
            w_blank_idx=self.w_dict.unk(),
            f_blank_idx=self.f_dict.unk(),
        )

        noisy_src_lengths, sort_order = noisy_src_lengths.sort(descending=True)

        return noisy_src_tokens[:, sort_order], noisy_src_fields[:, sort_order], noisy_src_lengths
