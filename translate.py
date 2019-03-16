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
import subprocess
import os
import pathlib
import multiprocessing as mp
import re


# BLEU_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
# BLEU_SCRIPT_PATH = os.path.join(TOOLS_PATH, 'mosesdecoder/scripts/generic/multi-bleu.perl')


def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(ref) and os.path.isfile(hyp)
    command = './multi-bleu.perl' + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith('BLEU'):
        # return float(result[7:result.index(',')])
        return result
    else:
        print('Impossible to parse BLEU score! "%s"' % result)
        # logger.warning('Impossible to parse BLEU score! "%s"' % result)
        return -1


def load_model(model, device): 
    translator = torch.load(model)
    translator.device = device
    translator.encoder_word_embeddings.to(device)
    translator.decoder_word_embeddings.to(device)
    translator.encoder_field_embeddings.to(device)
    translator.decoder_field_embeddings.to(device)
    translator.generator.to(device)
    translator.encoder.to(device)
    translator.decoder.to(device)
    translator.word_criterion.to(device)
    translator.field_criterion.to(device)

    return translator


def trans(args, model, is_cpu, que):
    if is_cpu:
        device = 'cpu'
    else:
        device = que.get()

    pid = os.getpid()

    model_it = re.search(r".it(?P<it>[\d]*)", str(model)).group('it')
    args.output = args.output + '.' + model_it

    print("[PID %d | it %s] Start evaluation model %s on device %s" % (pid, model_it, str(model), device))


    translator = load_model(model, device)
    print("[PID %d | it %s] Verify device %s" % (pid, model_it, translator.device))
    
    args.output = args.output + ''

    with ExitStack() as stack:
        fin_content = stack.enter_context(open(args.input + '.content', encoding=args.encoding, errors='surrogateescape'))
        fin_labels = stack.enter_context(open(args.input + '.labels', encoding=args.encoding, errors='surrogateescape'))
        fout_content = stack.enter_context(open(args.output + '.content', mode='w', encoding=args.encoding, errors='surrogateescape'))
        fout_labels = stack.enter_context(open(args.output + '.labels', mode='w', encoding=args.encoding, errors='surrogateescape'))

        bytes_read = 0
        total_bytes = os.path.getsize(args.input + '.content')
        target_bytes = 0
        end = False

        while not end:
            content_batch = []
            labels_batch = []
            while len(content_batch) < args.batch_size and not end:
                content = fin_content.readline()
                labels = fin_labels.readline()
                content_ids = [int(idstr) for idstr in content.strip().split()]
                labels_ids = [int(idstr) for idstr in labels.strip().split()]

                if bytes_read >= target_bytes:
                    print("[PID %d | it %s] progress %.3f" % (pid, model_it, 100.0 * (bytes_read / total_bytes)))
                    target_bytes += total_bytes // 20

                bytes_read += len(content)

                if not content:
                    end = True
                else:
                    content_batch.append(content_ids)
                    labels_batch.append(labels_ids)
            if args.beam_size <= 0 and len(content_batch) > 0:
                for idx, (w_translation, f_translation) in enumerate(zip(*translator.greedy(content_batch, labels_batch, train=False))):
                    w_str_trans = bpemb_en.decode_ids(w_translation)
                    f_str_trans = " ".join([translator.trg_field_dict.id2word[idx] for idx in f_translation])
                    fout_content.write(w_str_trans + '\n')
                    fout_labels.write(f_str_trans + '\n')
            elif len(content_batch) > 0:
                pass

    if not is_cpu:
        que.put(device)

    print("[PID %d | it %s] Evaluating BLEU" % (pid, model_it))
    result = eval_moses_bleu(args.ref + '.str.last.content', args.output + '.content')
    print("[PID %d | it %s] Done" % (pid, model_it))

    return int(model_it), str(model) + ': ' + result + '\n'


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Translate using a pre-trained model')
    parser.add_argument('model', help='a model previously trained with train.py')
    parser.add_argument('--batch_size', type=int, default=50, help='the batch size (defaults to 50)')
    parser.add_argument('--beam_size', type=int, default=0, help='the beam size (defaults to 12, 0 for greedy search)')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('-i', '--input', type=str, default='./data/processed_data/valid/valid.box',
                        help='the input file for translation')
    parser.add_argument('-o', '--output', type=str, default='./data/processed_data/valid/res.article',
                        help='the output file')
    parser.add_argument('--ref', type=str, default='./data/processed_data/valid/valid.article',
                        help='the reference file')
    parser.add_argument('--is_cpu', action='store_true')

    args = parser.parse_args()

    currDir = pathlib.Path('.')
    currPatt = "*MONO*"

    model_files = sorted([currFile for currFile in currDir.glob(currPatt)],
                         key=lambda x: int(re.search(r".it(?P<it>[\d]*)", str(x)).group('it')))

    if args.is_cpu:
        max_processes = 10
    else:
        max_processes = 4

    pool = mp.Pool(processes=max_processes)
    m = mp.Manager()
    q = m.Queue()

    if not args.is_cpu:
        q.put('cuda:0')
        q.put('cuda:1')
        q.put('cuda:5')
        q.put('cuda:6')

    results = [pool.apply_async(trans, args=(args, model, args.is_cpu, q)) for model in model_files]
    pool_outs = sorted([p.get() for p in results], key=lambda res: res[0])

    bleu_res_path = './data/processed_data/bleu_res_new.txt'
    with open(bleu_res_path, mode='w', encoding=args.encoding) as bleu_file:
        for pool_out in pool_outs:
            bleu_file.write(pool_out[1])


if __name__ == '__main__':
    main()
