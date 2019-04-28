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

import random
import torch
import torch.nn as nn
import logging
from src.data import LabelDict, BpeWordDict, UnsupervisedMTNoising
import copy

from torch.nn import functional as F

logger = logging.getLogger()


class Translator:
    def __init__(self, name, encoder_word_embeddings, decoder_word_embeddings,
                 encoder_field_embeddings, decoder_field_embeddings, generator, src_word_dict,
                 trg_word_dict, src_field_dict, trg_field_dict, src_type, trg_type, w_sos_id, bpemb_en,
                 encoder, decoder, discriminator=None, denoising=1, device='cpu',
                 noising_class=UnsupervisedMTNoising, **kwargs):
        self.name = name
        self.encoder_word_embeddings = encoder_word_embeddings
        self.decoder_word_embeddings = decoder_word_embeddings
        self.encoder_field_embeddings = encoder_field_embeddings
        self.decoder_field_embeddings = decoder_field_embeddings
        self.generator = generator
        self.src_word_dict: BpeWordDict = src_word_dict
        self.trg_word_dict: BpeWordDict = trg_word_dict
        self.src_field_dict: LabelDict = src_field_dict
        self.trg_field_dict: LabelDict = trg_field_dict
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.bpemb_en = bpemb_en
        self.noiser = noising_class(w_dict=src_word_dict, f_dict=src_field_dict, **kwargs)

        assert self.encoder.batch_first == self.decoder.batch_first

        self.batch_first = self.encoder.batch_first
        self.denoising = denoising
        self.device = device
        self.w_pad_id = self.src_word_dict.pad_index
        self.w_unk_id = self.src_word_dict.unk_index
        self.w_sos_id = w_sos_id
        self.w_eos_id = self.src_word_dict.eos_index
        self.f_pad_id = self.src_field_dict.pad_index
        self.f_unk_id = self.src_field_dict.unk_index
        self.f_null_id = self.src_field_dict.null_index

        self.src_type = src_type
        self.trg_type = trg_type

        word_class, field_classes = generator.output_classes()
        word_weight = torch.ones(word_class)
        field_weight = torch.ones(field_classes)
        word_weight[src_word_dict.pad_index] = 0
        field_weight[src_field_dict.pad_index] = 0
        self.word_criterion = nn.NLLLoss(word_weight, reduction='sum').to(device)
        self.field_criterion = nn.NLLLoss(field_weight, reduction='sum').to(device)
        logger.debug('word_criterion is running on cuda: %d', self.word_criterion.weight.is_cuda)
        logger.debug('field_criterion is running on cuda: %d', self.field_criterion.weight.is_cuda)

    def _train(self, mode):
        self.encoder_word_embeddings.train(mode)
        self.decoder_word_embeddings.train(mode)
        self.encoder_field_embeddings.train(mode)
        self.decoder_field_embeddings.train(mode)
        self.generator.train(mode)
        self.encoder.train(mode)
        self.decoder.train(mode)
        self.word_criterion.train(mode)
        self.field_criterion.train(mode)
        if self.discriminator is not None:
            self.discriminator.train(False)

    def add_control_sym(self, sentences, sentences_field, eos=False, sos=False):
        assert (eos ^ sos)

        def valid_and_return(sent, sent_field):
            assert len(sent) == len(sent_field)
            return sent, sent_field, len(sent)

        if sos:
            sent_field_len = [valid_and_return([self.w_sos_id] + sent, [self.f_null_id] + sent_field)
                              for sent, sent_field in zip(sentences, sentences_field)]
        else:
            sent_field_len = [valid_and_return(sent + [self.w_eos_id], sent_field + [self.f_null_id])
                              for sent, sent_field in zip(sentences, sentences_field)]

        sents, sents_field, lengths = zip(*sent_field_len)

        return list(sents), list(sents_field), list(lengths)

    def add_padding(self, sents, sents_field, max_length):
        sents = [s + [self.w_pad_id]*(max_length-len(s)) for s in sents]
        sents_field = [s + [self.f_pad_id] * (max_length - len(s)) for s in sents_field]

        return sents, sents_field

    def transpose_ids(self, sents, sents_field, max_length):
        sents = [[sents[i][j] for i in range(len(sents))] for j in range(max_length)]
        sents_field = [[sents_field[i][j] for i in range(len(sents_field))] for j in range(max_length)]

        return sents, sents_field

    # @staticmethod
    def add_noise(self, word_ids, field_ids):
        for i in range(len(word_ids)):
            length = len(word_ids[i])

            if length > 2:
                for it in range(length // 2):
                    j = random.randrange(length - 1)

                    word_ids[i][j], word_ids[i][j + 1] = word_ids[i][j + 1], word_ids[i][j]
                    field_ids[i][j], field_ids[i][j + 1] = field_ids[i][j + 1], field_ids[i][j]

        return word_ids, field_ids

    def preprocess_ids(self, sentences, sentences_field, train=False, eos=False, sos=False):
        if train and self.denoising == 1:
            # Add order noise
            sentences, sentences_field = self.add_noise(sentences, sentences_field)

        word_ids, field_ids, lengths = self.add_control_sym(sentences, sentences_field, eos, sos)
        max_length = max(lengths)

        # Padding
        word_ids, field_ids = self.add_padding(word_ids, field_ids, max_length)

        return word_ids, field_ids, lengths

    def encode(self, sentences, sentences_field, train=False):
        word_ids, field_ids, lengths = self.preprocess_ids(sentences, sentences_field, train=train, eos=True)

        with torch.no_grad():
            if not self.batch_first:
                var_wordids = torch.LongTensor(word_ids).transpose(1, 0).contiguous()
                var_fieldids = torch.LongTensor(field_ids).transpose(1, 0).contiguous()
                var_length = torch.LongTensor(lengths)
            else:
                var_wordids = torch.LongTensor(word_ids)
                var_fieldids = torch.LongTensor(field_ids)
                var_length = torch.LongTensor(lengths)

        if train and self.denoising == 2:
            var_wordids, var_fieldids, var_length = self.noiser.noising(var_wordids, var_fieldids, var_length)

        with torch.no_grad():
            var_wordids = torch.LongTensor(var_wordids).to(self.device)
            var_fieldids = torch.LongTensor(var_fieldids).to(self.device)
            var_length = torch.LongTensor(var_length).to(self.device)

        logger.debug('enc: word_ids are on cuda: %d', var_wordids.is_cuda)
        logger.debug('enc: field_ids are on cuda: %d', var_fieldids.is_cuda)

        hidden = self.encoder.initial_hidden(len(sentences)).to(self.device)
        logger.debug('hidden is on cuda: %d', hidden.is_cuda)

        hidden, context = self.encoder(word_ids=var_wordids, field_ids=var_fieldids, lengths=var_length,
                                       word_embeddings=self.encoder_word_embeddings,
                                       field_embeddings=self.encoder_field_embeddings, hidden=hidden)
        return hidden, context, var_length

    def mask(self, lengths):
        batch_size = len(lengths)
        max_length = max(lengths)
        if max_length == min(lengths):
            return None
        mask = torch.ByteTensor(batch_size, max_length).fill_(0)
        for i in range(batch_size):
            for j in range(lengths[i], max_length):
                mask[i, j] = 1
        return mask

    def decode(self, sentences, sentences_field, hidden, context, context_mask):
        batch_size = len(sentences)
        initial_output = self.decoder.initial_output(batch_size).to(self.device)
        logger.debug('initial_output is on cuda: %d', initial_output.is_cuda)

        in_word_ids, in_field_ids, lengths = self.preprocess_ids(sentences, sentences_field, sos=True)

        with torch.no_grad():
            if not self.batch_first:
                in_var_word_ids = torch.LongTensor(in_word_ids).transpose(1, 0).contiguous().to(self.device)
                in_var_field_ids = torch.LongTensor(in_field_ids).transpose(1, 0).contiguous().to(self.device)
            else:
                in_var_word_ids = torch.LongTensor(in_word_ids).to(self.device)
                in_var_field_ids = torch.LongTensor(in_field_ids).to(self.device)

            logger.debug('dec: word_ids are on cuda: %d', in_var_word_ids.is_cuda)
            logger.debug('dec: field_ids are on cuda: %d', in_var_field_ids.is_cuda)

        word_logprobs, field_logprobs, hidden, _ = self.decoder(in_var_word_ids, in_var_field_ids, lengths,
                                                                self.decoder_word_embeddings,
                                                                self.decoder_field_embeddings, hidden, context,
                                                                context_mask, initial_output, self.generator)

        return word_logprobs, field_logprobs, hidden

    def greedy(self, sentences, field_sentences, max_ratio=2, train=False):
        self._train(train)
        assert len(sentences) == len(field_sentences)
        for sent, sent_field in zip(sentences, field_sentences):
            assert len(sent) == len(sent_field)

        hidden, context, context_lengths = self.encode(sentences, field_sentences, train)
        logger.debug('hidden, context are on cuda: %d, %d', hidden.is_cuda, context.is_cuda)

        context_mask = self.mask(context_lengths)
        if context_mask is not None:
            context_mask = context_mask.to(self.device)
            logger.debug('context_mask is on cuda: %d', context_mask.is_cuda)

        word_translations = [[] for _ in sentences]
        field_translations = [[] for _ in field_sentences]
        prev_words = len(sentences)*[self.w_sos_id]
        prev_fields = len(sentences) * [self.f_null_id]
        pending = set(range(len(sentences)))
        output = self.decoder.initial_output(len(sentences)).to(self.device)

        attentions = [torch.empty_like(output) for _ in sentences]

        while len(pending) > 0:
            # Maybe add teacher forcing?
            with torch.no_grad():
                var_word = torch.LongTensor([prev_words]).to(self.device)
                var_field = torch.LongTensor([prev_fields]).to(self.device)
                logger.debug('greedy: word_ids are on cuda: %d', var_word.is_cuda)
                logger.debug('greedy: field_ids are on cuda: %d', var_field.is_cuda)

            word_logprobs, field_logprobs, hidden, output = self.decoder(var_word, var_field, len(sentences)*[1],
                                                                         self.decoder_word_embeddings,
                                                                         self.decoder_field_embeddings, hidden,
                                                                         context, context_mask, output,
                                                                         self.generator)

            logger.debug('greedy: word_logprobs is on cuda: %d', word_logprobs.is_cuda)
            logger.debug('greedy: field_logprobs is on cuda: %d', field_logprobs.is_cuda)
            logger.debug('greedy: hidden are is cuda: %d', hidden.is_cuda)
            logger.debug('greedy: output are is cuda: %d', output.is_cuda)

            prev_words = word_logprobs.max(dim=2)[1].squeeze().data.cpu().numpy().tolist()
            prev_fields = field_logprobs.max(dim=2)[1].squeeze().data.cpu().numpy().tolist()
            for i in pending.copy():
                if prev_words[i] == self.w_eos_id:
                    pending.discard(i)
                else:
                    word_translations[i].append(prev_words[i])
                    field_translations[i].append(prev_fields[i])
                    attentions[i] = output
                    if len(word_translations[i]) >= max_ratio*len(sentences[i]):
                        pending.discard(i)
        return word_translations, field_translations, attentions

    def score(self, src_word, trg_word, src_field, trg_field, print_dbg=False, train=False):
        self._train(train)

        # Check batch sizes
        if len(src_word) != len(trg_word) != len(src_field) != len(trg_field):
            raise Exception('Sentence and hypothesis lengths do not match')

        # Encode
        if print_dbg:
            wsrc_dbg = self.bpemb_en.decode_ids(src_word[0])
            fsrc_dbg = " ".join([self.trg_field_dict.id2word[idx] for idx in src_field[0]])

        # tmp_src_word = copy.deepcopy(src_word)
        # tmp_src_field = copy.deepcopy(src_field)
        hidden, context, context_lengths = self.encode(src_word, src_field, train)
        context_mask = self.mask(context_lengths)
        if context_mask is not None:
            context_mask = context_mask.to(self.device)
            logger.debug('score: h, c, c_m are on cuda: %d, %d, %d', hidden.is_cuda, context.is_cuda, context_mask.is_cuda)
        else:
            logger.debug('score: h, c are on cuda: %d, %d', hidden.is_cuda, context.is_cuda)

        if self.discriminator is not None:
            predictions = self.discriminator(hidden.view(-1, hidden.size(-1)))
            real_idx = 1 if self.src_type == 'text' else 0
            fake_y = torch.LongTensor(predictions.size(0)).fill_(1 - real_idx)
            fake_y = fake_y.to(self.device)
            dis_loss = F.cross_entropy(predictions, fake_y)
        else:
            dis_loss = torch.tensor(0.0)

        # Decode
        word_logprobs, field_logprobs, hidden = self.decode(trg_word, trg_field, hidden, context,
                                                            context_mask)

        # Compute loss
        out_word_ids, out_field_ids, lengths = self.preprocess_ids(trg_word, trg_field, eos=True, sos=False)

        with torch.no_grad():
            if not self.batch_first:
                out_word_ids_var = torch.LongTensor(out_word_ids).transpose(1, 0).contiguous().to(self.device)
                out_field_ids_var = torch.LongTensor(out_field_ids).transpose(1, 0).contiguous().to(self.device)
            else:
                out_word_ids_var = torch.LongTensor(out_word_ids).to(self.device)
                out_field_ids_var = torch.LongTensor(out_field_ids).to(self.device)

            # if not self.batch_first:
            #     out_word_ids_var = out_word_ids_var.transpose(1, 0)
            #     out_field_ids_var = out_field_ids_var.transpose(1, 0)

            logger.debug('score: word_ids are on cuda: %d', out_word_ids_var.is_cuda)
            logger.debug('score: field_ids are on cuda: %d', out_field_ids_var.is_cuda)

        # if print_dbg and self.name == 'src2trg':
        if print_dbg:
            test_exp_sent = out_word_ids_var.t()[0][0:lengths[0]]
            test_exp_field = out_field_ids_var.t()[0].data.cpu().numpy().tolist()
            test_res_sent = word_logprobs.max(dim=2)[1].t()[0]
            test_res_field = field_logprobs.max(dim=2)[1].t()[0].data.cpu().numpy().tolist()
            src_sent_name = "[" + self.name + ":" + "IN|CONTENT" + "] "
            src_field_name = "[" + self.name + ":" + "IN|LABELS" + "] "
            exp_sent_name = "[" + self.name + ":" + "OUT_EXP|CONTENT" + "] "
            exp_field_name = "[" + self.name + ":" + "OUT_EXP|LABELS" + "] "
            res_sent_name = "[" + self.name + ":" + "OUT_RES|CONTENT" + "] "
            res_field_name = "[" + self.name + ":" + "OUT_RES|LABELS" + "] "
            try:
                if max(lengths) - lengths[0]:
                    temp = (max(lengths) - lengths[0]) * [self.trg_word_dict.id2word[self.w_pad_id]]
                    temp = " " + " ".join(temp)
                else:
                    temp = ""
                print(src_sent_name + wsrc_dbg)
                print(exp_sent_name + self.bpemb_en.decode_ids(test_exp_sent) + temp)
                print(res_sent_name + self.bpemb_en.decode_ids(test_res_sent))
                print(src_field_name + fsrc_dbg)
                print(exp_field_name + " ".join([self.trg_field_dict.id2word[idx] for idx in test_exp_field]))
                print(res_field_name + " ".join([self.trg_field_dict.id2word[idx] for idx in test_res_field]))
                print('\n')
            except:
                print("An exception occurred")

        word_loss = self.word_criterion(word_logprobs.view(-1, word_logprobs.size()[-1]), out_word_ids_var.view(-1))
        field_loss = self.field_criterion(field_logprobs.view(-1, field_logprobs.size()[-1]), out_field_ids_var.view(-1))

        return word_loss, field_loss, dis_loss
