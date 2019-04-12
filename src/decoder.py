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

import torch
import torch.nn as nn
from src.attention import GlobalAttention


class RNNAttentionDecoder(nn.Module):
    def __init__(self, word_embedding_size, field_embedding_size, hidden_size, layers=1, dropout=0, input_feeding=True,
                 batch_first=False):
        super(RNNAttentionDecoder, self).__init__()
        self.batch_first = batch_first
        self.layers = layers
        self.hidden_size = hidden_size
        # self.special_embeddings = nn.Embedding(data.SPECIAL_SYMBOLS+1, word_embedding_size, padding_idx=0)
        self.attention = GlobalAttention(hidden_size, alignment_function='general')
        self.input_feeding = input_feeding
        self.input_size = word_embedding_size + field_embedding_size
        if input_feeding:
            self.input_size += hidden_size

        self.stacked_rnn = StackedGRU(self.input_size, hidden_size, layers=layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, word_ids, field_ids, lengths, word_embeddings, field_embeddings, hidden, context, context_mask,
                prev_output, generator):
        w_embeddings = word_embeddings(word_ids)  # + self.special_embeddings(data.special_ids(word_ids))
        f_embeddings = field_embeddings(field_ids)  # ???
        output = prev_output
        word_scores = []
        field_scores = []
        for w_emb, f_emb in zip(w_embeddings.split(1), f_embeddings.split(1)):
            if self.input_feeding:
                input = torch.cat([w_emb.squeeze(0), f_emb.squeeze(0), output], 2)
            else:
                input = torch.cat([w_emb.squeeze(0), f_emb.squeeze(0)], 1)
            output, hidden = self.stacked_rnn(input, hidden)
            output = self.attention(output, context, context_mask)
            output = self.dropout(output)
            word_score, field_score = generator(output)
            word_scores.append(word_score)
            field_scores.append(field_score)
        return torch.stack(word_scores), torch.stack(field_scores), hidden, output

    def initial_output(self, batch_size):
        with torch.no_grad():
            init_output = torch.zeros(batch_size, self.hidden_size)

        return init_output


# Based on OpenNMT-py
class StackedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, layers, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = layers
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
        h_1 = torch.stack(h_1)
        return input, h_1
