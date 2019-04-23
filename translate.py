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


def trans(args, input_filepath, output_dir, ref_filepath, model, bpemb_en, que):
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
                for idx, (w_trans, f_trans) in enumerate(zip(*translator.greedy(content_batch, labels_batch, train=False))):
                    w_str_trans = bpemb_en.decode_ids(w_trans)
                    f_str_trans = " ".join([translator.trg_field_dict.id2word[field_idx] for field_idx in f_trans])
                    try:
                        fout_content.write(w_str_trans + '\n')
                        fout_labels.write(f_str_trans + '\n')
                    except:
                        print("Error in greedy decoding")
                        print(w_str_trans)
                        print(f_str_trans)
                        fout_content.write('\n')
                        fout_labels.write('\n')
            elif len(content_batch) > 0:
                pass

    que.put(device)

    print("[DEVICE %s | PID %d | it %s] Evaluating BLEU" % (device, pid, model_it))
    result = eval_moses_bleu(ref_filepath, output_filepath + '.content')
    print("[DEVICE %s | PID %d | it %s] %s" % (device, pid, model_it, result))
    print("[DEVICE %s | PID %d | it %s] Done" % (device, pid, model_it))

    return int(model_it), str(model) + ': ' + result + '\n'


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Translate using a pre-trained model')
    parser.add_argument('--model', type=str, default='', help='a .pth model file previously trained with train.py')
    parser.add_argument('--model_list', type=str, default='', help='a list of (space separated) model files')
    parser.add_argument('--batch_size', type=int, default=50, help='the batch size (defaults to 50)')
    parser.add_argument('--beam_size', type=int, default=0, help='the beam size (defaults to 12, 0 for greedy search)')
    parser.add_argument('--encoding', default='utf-8',
                        help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--testset_path', type=str, default='./data/processed_data/valid',
                        help='test data path')
    parser.add_argument('--device_list', type=str, default='cuda:0')
    parser.add_argument('--prefix', type=str, default='MONO')
    parser.add_argument('--train_corpus_mode', type=str, default='MONO', help='MONO/PARA')
    parser.add_argument('--direction', type=str, default='table2text', help='table2text/text2table')

    args = parser.parse_args()

    metadata_path = config.PRC_TRAIN_DATA_PATH + '/metadata.bin'
    assert os.path.isfile(metadata_path)

    metadata = torch.load(metadata_path)
    bpemb_en = metadata.init_bpe_module()

    currDir = pathlib.Path('.')
    currPatt = args.prefix + ".it*.src2trg*"

    assert os.path.isdir(args.testset_path), "{} is not a directory".format(args.testset_path)
    test_basename = os.path.basename(args.testset_path)
    test_basedir = os.path.abspath(args.testset_path)

    if args.direction == 'table2text':
        in_suffix = '.box'
        out_suffix = '.article'
    else:
        assert args.direction == 'text2table'
        in_suffix = '.article'
        out_suffix = '.box'

    input_filepath = os.path.join(test_basedir, test_basename + in_suffix)
    ref_path = os.path.join(test_basedir,  test_basename + out_suffix)

    if args.train_corpus_mode == 'MONO':
        input_filepath += '.mono'
        ref_path += '.mono'

    ref_string_path = ref_path + '.str.content'

    output_dir = os.path.join(test_basedir, os.path.join('translations', args.prefix))
    safe_mkdir(local_path_to(output_dir))

    if not os.path.isfile(ref_string_path):
        print("Creating ref file... [%s] [%s]" % (ref_path + '.content', ref_string_path))

        with ExitStack() as stack:

            fref_content = stack.enter_context(
                open(ref_path + '.content', encoding=args.encoding, errors='surrogateescape'))
            fref_str_content = stack.enter_context(
                open(ref_string_path, mode='w', encoding=args.encoding, errors='surrogateescape'))

            for line in fref_content:
                ref_ids = [int(idstr) for idstr in line.strip().split()]
                ref_str = bpemb_en.decode_ids(ref_ids)
                fref_str_content.write(ref_str + '\n')

        print("Ref file created!")

    if args.model == '':
        model_list_pathname = os.path.abspath(args.model_list)
        if os.path.isfile(model_list_pathname):
            with open(model_list_pathname, encoding=args.encoding, errors='surrogateescape') as model_list_file:
                model_files = model_list_file.readlines()
                assert len(model_files) == 1, "file should have a single line with space separated model files"
                model_files = model_files.split()
        else:
            model_files = sorted([currFile for currFile in currDir.glob(currPatt)],
                                 key=lambda x: int(re.search(r".it(?P<it>[\d]*)", str(x)).group('it')),reverse=True)
    else:
        model_files = [args.model]

    print("Number of models: %d" % (len(model_files)))

    device_list = args.device_list.split()
    max_processes = len(device_list)

    pool = mp.Pool(processes=max_processes)
    m = mp.Manager()
    q = m.Queue()

    if args.model == '':
        for dev in device_list:
            q.put(dev)

        results = [pool.apply_async(trans, args=(args, input_filepath, output_dir, ref_string_path,
                                                 model, bpemb_en, q)) for model in model_files]
        pool_outs = sorted([p.get() for p in results], key=lambda res: res[0])

        bleu_res_path = './data/processed_data/' + args.prefix + '_bleu_res_new.txt'

        with open(bleu_res_path, mode='w', encoding=args.encoding) as bleu_file:
            for pool_out in pool_outs:
                bleu_file.write(pool_out[1])
    else:
        q.put('cuda:0')
        bleu_res = trans(args, input_filepath, output_dir, ref_string_path, model_files[0], bpemb_en, q)
        print(bleu_res)


if __name__ == '__main__':
    main()
