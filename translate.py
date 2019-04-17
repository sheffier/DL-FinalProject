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
from nltk.translate.bleu_score import sentence_bleu
import subprocess
import os
import pathlib
import multiprocessing as mp
import re
import config
from src.utils import local_path_to, safe_mkdir
from preprocess import PreprocessMetadata


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
    translator = torch.load(model, map_location=device)
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


def trans(args, input_filepath, output_dir, ref_filepath, model, bpemb_en, is_cpu, que):
    if is_cpu:
        device = 'cpu'
    else:
        device = que.get()

    pid = os.getpid()

    model_it = re.search(r".it(?P<it>[\d]*)", str(model)).group('it')
    output_filepath = os.path.join(output_dir, model_it)

    print("[DEVICE %s | PID %d | it %s] Start evaluation model %s on device %s" %
          (device, pid, model_it, str(model), device))

    translator = load_model(model, device)

    print("[DEVICE %s | PID %d | it %s] Verify device %s" % (device, pid, model_it, translator.device))

    output_filepath = output_filepath + ''

    with ExitStack() as stack:
        fin_content = stack.enter_context(open(input_filepath + '.content',
                                               encoding=args.encoding, errors='surrogateescape'))
        fin_labels = stack.enter_context(open(input_filepath + '.labels',
                                              encoding=args.encoding, errors='surrogateescape'))
        fout_content = stack.enter_context(open(output_filepath + '.content',
                                                mode='w', encoding=args.encoding, errors='surrogateescape'))
        fout_labels = stack.enter_context(open(output_filepath + '.labels',
                                               mode='w', encoding=args.encoding, errors='surrogateescape'))

        bytes_read = 0
        total_bytes = os.path.getsize(input_filepath + '.content')
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
                    print("[DEVICE %s | PID %d | it %s] progress %.3f" %
                          (device, pid, model_it, 100.0 * (bytes_read / total_bytes)))
                    target_bytes += total_bytes // 20

                bytes_read += len(content)

                if not content:
                    end = True
                else:
                    content_batch.append(content_ids)
                    labels_batch.append(labels_ids)

            if args.beam_size <= 0 and len(content_batch) > 0:
                for idx, translation in enumerate(zip(*translator.greedy(content_batch, labels_batch, train=False))):
                    w_trans, f_trans = translation
                    w_str_trans = bpemb_en.decode_ids(w_trans)
                    f_str_trans = " ".join([translator.trg_field_dict.id2word[field_idx] for field_idx in f_trans])
                    fout_content.write(w_str_trans + '\n')
                    fout_labels.write(f_str_trans + '\n')
            elif len(content_batch) > 0:
                pass

    if not is_cpu:
        que.put(device)

    print("[DEVICE %s | PID %d | it %s] Evaluating BLEU" % (device, pid, model_it))
    result = eval_moses_bleu(ref_filepath + '.str.content', output_filepath + '.content')
    print("[DEVICE %s | PID %d | it %s] Done" % (device, pid, model_it))

    return int(model_it), str(model) + ': ' + result + '\n'


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Translate using a pre-trained model')
    parser.add_argument('--model', type=str, default='', help='a model previously trained with train.py')
    parser.add_argument('--batch_size', type=int, default=50, help='the batch size (defaults to 50)')
    parser.add_argument('--beam_size', type=int, default=0, help='the beam size (defaults to 12, 0 for greedy search)')
    parser.add_argument('--encoding', default='utf-8',
                        help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--testset_path', type=str, default='./data/processed_data/valid',
                        help='test data path')
    parser.add_argument('--is_cpu', action='store_true')
    parser.add_argument('--prefix', type=str, default='MONO')

    args = parser.parse_args()

    metadataPath = config.PRC_TRAIN_DATA_PATH + '/metadata.bin'
    assert os.path.isfile(metadataPath)

    metadata = torch.load(metadataPath)
    bpemb_en = metadata.init_bpe_module()

    currDir = pathlib.Path('.')
    currPatt = "*" + args.prefix + ".it*.src2trg*"

    assert os.path.isdir(args.testset_path), "{} is not a directory".format(args.testset_path)
    test_basename = os.path.basename(args.testset_path)
    test_basedir = os.path.abspath(args.testset_path)

    input_filepath = os.path.join(test_basedir, test_basename + '.box')
    ref_path = os.path.join(test_basedir,  test_basename + '.article')
    ref_string_path = ref_path + '.str.content'

    output_dir = os.path.join(test_basedir, os.path.join('translations', args.prefix))
    safe_mkdir(local_path_to(output_dir))

    if not os.path.isfile(ref_string_path):
        print("Creating ref file...")

        with ExitStack() as stack:
            fref_content = stack.enter_context(
                open(ref_path + '.content', encoding=args.encoding, errors='surrogateescape'))
            fref_str_content = stack.enter_context(
                open(ref_path + '.str.content', mode='w', encoding=args.encoding, errors='surrogateescape'))

            for line in fref_content:
                ref_ids = [int(idstr) for idstr in line.strip().split()]
                ref_str = bpemb_en.decode_ids(ref_ids)
                fref_str_content.write(ref_str + '\n')

        print("Ref file created!")

    if args.model == '':
        model_files = sorted([currFile for currFile in currDir.glob(currPatt)],
                             key=lambda x: int(re.search(r".it(?P<it>[\d]*)", str(x)).group('it')))
    else:
        model_files = [args.model]

    print("Number of models: %d" % (len(model_files)))

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
        q.put('cuda:2')
        q.put('cuda:3')

    results = [pool.apply_async(trans, args=(args, input_filepath, output_dir, ref_string_path,
                                             model, bpemb_en, args.is_cpu, q)) for model in model_files]
    pool_outs = sorted([p.get() for p in results], key=lambda res: res[0])

    bleu_res_path = './data/processed_data/' + args.prefix + '_bleu_res_new.txt'

    with open(bleu_res_path, mode='w', encoding=args.encoding) as bleu_file:
        for pool_out in pool_outs:
            bleu_file.write(pool_out[1])


if __name__ == '__main__':
    main()
