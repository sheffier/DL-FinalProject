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

from src import data

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import bpemb_en


class EmbeddingGenerator(nn.Module):
    def __init__(self, hidden_size, word_embedding_size, field_embedding_size):
        super(EmbeddingGenerator, self).__init__()
        self.hidden2embedding = nn.Linear(hidden_size, word_embedding_size + field_embedding_size)
        # self.special_out = nn.Linear(word_embedding_size, data.SPECIAL_SYMBOLS, bias=False)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, hidden, embeddings):
        emb = self.hidden2embedding(hidden)
        # word_scores = F.linear(emb, embeddings.weight[1:, :])
        # special_scores = self.special_out(emb)
        # scores = torch.cat((special_scores, word_scores), dim=1)

        scores = F.linear(emb, embeddings.weight[1:, :])

        return self.logsoftmax(scores)

    def output_classes(self):
        return None


class WrappedEmbeddingGenerator(nn.Module):
    def __init__(self, embedding_generator, embeddings):
        super(WrappedEmbeddingGenerator, self).__init__()
        self.embedding_generator = embedding_generator
        self.embeddings = embeddings

    def forward(self, hidden):
        return self.embedding_generator(hidden, self.embeddings)

    def output_classes(self):
        return self.embeddings.weight.data.size()[0] + data.SPECIAL_SYMBOLS - 1


class LinearGenerator(nn.Module):
    def __init__(self, hidden_size, word_vocab_size, field_vocab_size, bias=True):
        super(LinearGenerator, self).__init__()
        self.field_scores = nn.Linear(hidden_size, field_vocab_size, bias=bias)
        self.word_scores = nn.Linear(hidden_size, word_vocab_size, bias=bias)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, hidden):
        return self.logsoftmax(self.word_scores(hidden)), self.logsoftmax(self.field_scores(hidden))

    def output_classes(self):
        return self.word_scores.weight.size()[0], self.field_scores.weight.size()[0]
