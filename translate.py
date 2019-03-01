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

import argparse
import sys
import torch
from contextlib import ExitStack
from src.data import bpemb_en
from nltk.translate.bleu_score import sentence_bleu


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Translate using a pre-trained model')
    parser.add_argument('model', help='a model previously trained with train.py')
    parser.add_argument('--batch_size', type=int, default=50, help='the batch size (defaults to 50)')
    # parser.add_argument('--beam_size', type=int, default=12, help='the beam size (defaults to 12, 0 for greedy search)')
    parser.add_argument('--beam_size', type=int, default=0, help='the beam size (defaults to 12, 0 for greedy search)')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    # parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the input file (defaults to stdin)')
    # parser.add_argument('-o', '--output', default=sys.stdout.fileno(), help='the output file (defaults to stdout)')
    parser.add_argument('-i', '--input', type=str, default='./data/processed_data/valid/valid.box',
                        help='the input file for translation')
    parser.add_argument('-o', '--output', type=str, default='./data/processed_data/valid/res.article',
                        help='the output file')
    parser.add_argument('--ref', type=str, default='./data/processed_data/valid/valid.article',
                        help='the reference file')

    args = parser.parse_args()

    # Load model
    translator = torch.load(args.model)

    # Translate sentences
    end = False

    # fin = open(args.input, encoding=args.encoding, errors='surrogateescape')
    # fout = open(args.output, mode='w', encoding=args.encoding, errors='surrogateescape')

    while not end:
        with ExitStack() as stack:
            fin_content = stack.enter_context(open(args.input + '.content', encoding=args.encoding, errors='surrogateescape'))
            fin_labels = stack.enter_context(open(args.input + '.labels', encoding=args.encoding, errors='surrogateescape'))
            fout_content = stack.enter_context(open(args.output + '.content', mode='w', encoding=args.encoding, errors='surrogateescape'))
            fout_labels = stack.enter_context(open(args.output + '.labels', mode='w', encoding=args.encoding, errors='surrogateescape'))
            fref_content = stack.enter_context(open(args.ref + '.content', encoding=args.encoding, errors='surrogateescape'))

            content_batch = []
            labels_batch = []
            ref_batch = []
            while len(content_batch) < args.batch_size and not end:
                content = fin_content.readline()
                labels = fin_labels.readline()
                ref = fref_content.readline()
                content_ids = [int(idstr) for idstr in content.strip().split()]
                labels_ids = [int(idstr) for idstr in labels.strip().split()]
                ref_ids = [int(idstr) for idstr in ref.strip().split()]

                if not content:
                    end = True
                else:
                    content_batch.append(content_ids)
                    labels_batch.append(labels_ids)
                    ref_batch.append(ref_ids)
            if args.beam_size <= 0 and len(content_batch) > 0:
                for idx, (w_translation, f_translation) in enumerate(zip(*translator.greedy(content_batch, labels_batch, train=False))):
                    w_str_trans = bpemb_en.decode_ids(w_translation)
                    f_str_trans = " ".join([translator.trg_field_dict.id2word[idx] for idx in f_translation])
                    ref_str = bpemb_en.decode_ids(ref_batch[idx])
                    fout_content.write(w_str_trans + '\n')
                    fout_labels.write(f_str_trans + '\n')
                    print(w_str_trans.encode('utf-8'))
                    print(ref_str.encode('utf-8'))
                    print('BLEU-4: %f' % 100 * sentence_bleu([ref_str.split()], w_str_trans.split()))
            elif len(content_batch) > 0:
                pass
                # for translation in translator.beam_search(batch, train=False, beam_size=args.beam_size):
                #     print(translation, file=fout)

            # fout.flush()

    # fin.close()
    # fout.close()


if __name__ == '__main__':
    main()
