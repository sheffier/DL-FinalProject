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


class RNNEncoder(nn.Module):
    def __init__(self, word_embedding_size, field_embedding_size, hidden_size, bidirectional=False, layers=1,
                 dropout=0, batch_first=False):
        super(RNNEncoder, self).__init__()
        if bidirectional and hidden_size % 2 != 0:
            raise ValueError('The hidden dimension must be even for bidirectional encoders')
        self.directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.layers = layers
        self.hidden_size = hidden_size // self.directions
        # self.special_embeddings = nn.Embedding(data.SPECIAL_SYMBOLS+1, word_embedding_size, padding_idx=0)
        self.rnn = nn.GRU(word_embedding_size + field_embedding_size, self.hidden_size, bidirectional=bidirectional,
                          num_layers=layers, dropout=dropout, batch_first=batch_first)

    def forward(self, word_ids, field_ids, lengths, word_embeddings, field_embeddings, hidden):
        sorted_lengths = sorted(lengths, reverse=True)
        is_sorted = sorted_lengths == lengths
        is_varlen = sorted_lengths[0] != sorted_lengths[-1]
        if not is_sorted:
            true2sorted = sorted(range(len(lengths)), key=lambda x: -lengths[x])
            sorted2true = sorted(range(len(lengths)), key=lambda x: true2sorted[x])
            word_ids = torch.stack([word_ids[:, i] for i in true2sorted], dim=1)
            field_ids = torch.stack([field_ids[:, i] for i in true2sorted], dim=1)
            lengths = [lengths[i] for i in true2sorted]
        w_embeddings = word_embeddings(word_ids)  # + self.special_embeddings(data.special_ids(word_ids))
        f_embeddings = field_embeddings(field_ids)  # + self.special_embeddings(data.special_ids(word_ids))
        embeddings = torch.cat([w_embeddings, f_embeddings], 2)   # ????
        if is_varlen:
            embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=self.batch_first)

        output, hidden = self.rnn(embeddings, hidden)
        if self.bidirectional:
            hidden = torch.stack([torch.cat((hidden[2*i], hidden[2*i+1]), dim=1) for i in range(self.layers)])
        if is_varlen:
            output = nn.utils.rnn.pad_packed_sequence(output, batch_first=self.batch_first)[0]
        if not is_sorted:
            hidden = torch.stack([hidden[:, i, :] for i in sorted2true], dim=1)
            output = torch.stack([output[:, i, :] for i in sorted2true], dim=1)
        return hidden, output

    def initial_hidden(self, batch_size):
        with torch.no_grad():
            init_hidden = torch.zeros(self.layers*self.directions, batch_size, self.hidden_size)

        return init_hidden
