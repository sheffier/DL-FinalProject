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
import numpy as np
import sys
import time
import logging
import torch
import os
import datetime
import threading

import src.data as data
from src.encoder import RNNEncoder
from src.decoder import RNNAttentionDecoder
from src.discriminator import Discriminator
from src.generator import *
from src.translator import Translator
from src.data import BpeWordDict, LabelDict
from torch import nn
from contextlib import ExitStack
from preprocess import preprocess
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from src.utils import get_num_lines
from translate import calc_bleu


def main_train():
    # Build argument parser
    parser = argparse.ArgumentParser(description='Train a table to text model')

    # Training corpus
    corpora_group = parser.add_argument_group('training corpora',
                                              'Corpora related arguments; specify either unaligned or'
                                              ' aligned training corpora')
    # "Languages (type,path)"
    corpora_group.add_argument('--src_corpus_params', type=str,
                               default='table, ./data/processed_data/train/train.box',
                               help='the source unaligned corpus (type,path). Type = text/table')
    corpora_group.add_argument('--trg_corpus_params', type=str,
                               default='text, ./data/processed_data/train/train.article',
                               help='the target unaligned corpus (type,path). Type = text/table')
    corpora_group.add_argument('--src_para_corpus_params', type=str, default='',
                               help='the source corpus of parallel data(type,path). Type = text/table')
    corpora_group.add_argument('--trg_para_corpus_params', type=str, default='',
                               help='the target corpus of parallel data(type,path). Type = text/table')
    # Maybe add src/target type (i.e. text/table)
    corpora_group.add_argument('--corpus_mode', type=str, default='mono',
                               help='training mode: "mono" (unsupervised) / "para" (supervised)')

    corpora_group.add_argument('--max_sentence_length', type=int, default=50,
                               help='the maximum sentence length for training (defaults to 50)')
    corpora_group.add_argument('--cache', type=int, default=100000,
                               help='the cache size (in sentences) for corpus reading (defaults to 1000000)')

    # Embeddings/vocabulary
    embedding_group = parser.add_argument_group('embeddings',
                                                'Embedding related arguments; either give pre-trained embeddings,'
                                                ' or a vocabulary and embedding dimensionality to'
                                                ' randomly initialize them')
    embedding_group.add_argument('--metadata_path', type=str, default='', required=True,
                                 help='Path for bin file created in pre-processing phase, '
                                      'containing BPEmb related metadata.')

    # Architecture
    architecture_group = parser.add_argument_group('architecture', 'Architecture related arguments')
    architecture_group.add_argument('--layers', type=int, default=2,
                                    help='the number of encoder/decoder layers (defaults to 2)')
    architecture_group.add_argument('--hidden', type=int, default=600,
                                    help='the number of dimensions for the hidden layer (defaults to 600)')
    architecture_group.add_argument('--dis_hidden', type=int, default=150,
                                    help='Number of dimensions for the discriminator hidden layers')
    architecture_group.add_argument('--n_dis_layers', type=int, default=2,
                                    help='Number of discriminator layers')
    architecture_group.add_argument('--disable_bidirectional', action='store_true',
                                    help='use a single direction encoder')
    architecture_group.add_argument('--disable_backtranslation', action='store_true', help='disable backtranslation')
    architecture_group.add_argument('--disable_field_loss', action='store_true', help='disable backtranslation')
    architecture_group.add_argument('--disable_discriminator', action='store_true', help='disable discriminator')
    architecture_group.add_argument('--shared_enc', action='store_true', help='share enc for both directions')
    architecture_group.add_argument('--shared_dec', action='store_true', help='share dec for both directions')

    # Denoising
    denoising_group = parser.add_argument_group('denoising', 'Denoising related arguments')
    denoising_group.add_argument('--denoising_mode', type=int, default=1, help='0/1/2 = disabled/old/new')
    denoising_group.add_argument('--word_shuffle', type=int, default=3,
                                 help='shuffle words (only relevant in new mode)')
    denoising_group.add_argument('--word_dropout', type=float, default=0.1,
                                 help='randomly remove words (only relevant in new mode)')
    denoising_group.add_argument('--word_blank', type=float, default=0.2,
                                 help='randomly blank out words (only relevant in new mode)')

    # Optimization
    optimization_group = parser.add_argument_group('optimization', 'Optimization related arguments')
    optimization_group.add_argument('--batch', type=int, default=50, help='the batch size (defaults to 50)')
    optimization_group.add_argument('--learning_rate', type=float, default=0.0002,
                                    help='the global learning rate (defaults to 0.0002)')
    optimization_group.add_argument('--dropout', metavar='PROB', type=float, default=0.3,
                                    help='dropout probability for the encoder/decoder (defaults to 0.3)')
    optimization_group.add_argument('--param_init', metavar='RANGE', type=float, default=0.1,
                                    help='uniform initialization in the specified range (defaults to 0.1,  0 for module specific default initialization)')
    optimization_group.add_argument('--iterations', type=int, default=300000,
                                    help='the number of training iterations (defaults to 300000)')

    # Model saving
    saving_group = parser.add_argument_group('model saving', 'Arguments for saving the trained model')
    saving_group.add_argument('--save', metavar='PREFIX', help='save models with the given prefix')
    saving_group.add_argument('--save_interval', type=int, default=0, help='save intermediate models at this interval')

    # Logging/validation
    logging_group = parser.add_argument_group('logging', 'Logging and validation arguments')
    logging_group.add_argument('--log_interval', type=int, default=100, help='log at this interval (defaults to 1000)')
    logging_group.add_argument('--dbg_print_interval', type=int, default=1000,
                               help='log at this interval (defaults to 1000)')
    logging_group.add_argument('--src_valid_corpus', type=str, default='')
    logging_group.add_argument('--trg_valid_corpus', type=str, default='')
    logging_group.add_argument('--print_level', type=str, default='info', help='logging level [debug | info]')

    # Other
    misc_group = parser.add_argument_group('misc', 'Misc. arguments')
    misc_group.add_argument('--encoding', default='utf-8',
                            help='the character encoding for input/output (defaults to utf-8)')
    misc_group.add_argument('--cuda', type=str, default='cpu', help='device for training. default value: "cpu"')
    misc_group.add_argument('--bleu_device', type=str, default='',
                            help='device for calculating BLEU scores in case a validation dataset is given')

    # Parse arguments
    args = parser.parse_args()

    logger = logging.getLogger()
    if args.print_level == 'debug':
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    elif args.print_level == 'info':
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    elif args.print_level == 'warning':
        logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)

    # Validate arguments
    if args.src_corpus_params is None or args.trg_corpus_params is None:
        print("Must supply corpus")
        sys.exit(-1)

    args.src_corpus_params = args.src_corpus_params.split(',')
    args.trg_corpus_params = args.trg_corpus_params.split(',')
    assert len(args.src_corpus_params) == 2
    assert len(args.trg_corpus_params) == 2

    src_type, src_corpus_path = args.src_corpus_params
    trg_type, trg_corpus_path = args.trg_corpus_params

    src_type = src_type.strip()
    src_corpus_path = src_corpus_path.strip()
    trg_type = trg_type.strip()
    trg_corpus_path = trg_corpus_path.strip()

    assert src_type != trg_type
    assert (src_type in ['table', 'text']) and (trg_type in ['table', 'text'])

    corpus_size = get_num_lines(src_corpus_path + '.content')

    # Select device
    if torch.cuda.is_available():
        device = torch.device(args.cuda)
    else:
        device = torch.device('cpu')

    if args.bleu_device == '':
        args.bleu_device = device

    current_time = str(datetime.datetime.now().timestamp())
    run_dir = 'run_' + current_time + '/'
    train_log_dir = 'logs/train/' + run_dir + args.save
    valid_log_dir = 'logs/valid/' + run_dir + args.save

    train_writer = SummaryWriter(train_log_dir)
    valid_writer = SummaryWriter(valid_log_dir)

    # Create optimizer lists
    src2src_optimizers = []
    trg2trg_optimizers = []
    src2trg_optimizers = []
    trg2src_optimizers = []

    # Method to create a module optimizer and add it to the given lists
    def add_optimizer(module, directions=()):
        if args.param_init != 0.0:
            for param in module.parameters():
                param.data.uniform_(-args.param_init, args.param_init)
        optimizer = torch.optim.Adam(module.parameters(), lr=args.learning_rate)
        for direction in directions:
            direction.append(optimizer)
        return optimizer

    assert os.path.isfile(args.metadata_path)

    metadata = torch.load(args.metadata_path)
    bpemb_en = metadata.init_bpe_module()
    word_dict: BpeWordDict = torch.load(metadata.word_dict_path)
    field_dict: LabelDict = torch.load(metadata.field_dict_path)

    args.hidden = bpemb_en.dim + bpemb_en.dim // 2
    if not args.disable_bidirectional:
        args.hidden *= 2

    # Load embedding and/or vocab
    # word_dict = BpeWordDict.get(vocab=bpemb_en.words)
    w_sos_id = {'text': word_dict.bos_index, 'table': word_dict.sot_index}

    word_embeddings = nn.Embedding(len(word_dict), bpemb_en.dim, padding_idx=word_dict.pad_index)
    nn.init.normal_(word_embeddings.weight, 0, 0.1)
    nn.init.constant_(word_embeddings.weight[word_dict.pad_index], 0)
    with torch.no_grad():
        word_embeddings.weight[:bpemb_en.vs, :] = torch.from_numpy(bpemb_en.vectors)
    word_embedding_size = word_embeddings.weight.data.size()[1]
    word_embeddings = word_embeddings.to(device)
    word_embeddings.weight.requires_grad = False
    logger.debug('w_embeddings is running on cuda: %d', next(word_embeddings.parameters()).is_cuda)

    # field_dict: LabelDict = torch.load('./data/processed_data/train/field.dict')
    field_embeddings = nn.Embedding(len(field_dict), bpemb_en.dim // 2, padding_idx=field_dict.pad_index)
    nn.init.normal_(field_embeddings.weight, 0, 0.1)
    nn.init.constant_(field_embeddings.weight[field_dict.pad_index], 0)
    field_embedding_size = field_embeddings.weight.data.size()[1]
    field_embeddings = field_embeddings.to(device)
    field_embeddings.weight.requires_grad = True
    logger.debug('f_embeddings is running on cuda: %d', next(word_embeddings.parameters()).is_cuda)

    src_encoder_word_embeddings = word_embeddings
    trg_encoder_word_embeddings = word_embeddings
    src_encoder_field_embeddings = field_embeddings
    trg_encoder_field_embeddings = field_embeddings

    src_decoder_word_embeddings = word_embeddings
    trg_decoder_word_embeddings = word_embeddings
    src_decoder_field_embeddings = field_embeddings
    trg_decoder_field_embeddings = field_embeddings

    src_generator = LinearGenerator(args.hidden, len(word_dict), len(field_dict)).to(device)

    if args.shared_dec:
        trg_generator = src_generator
        add_optimizer(src_generator, (src2src_optimizers, trg2src_optimizers, trg2trg_optimizers, src2trg_optimizers))
    else:
        trg_generator = LinearGenerator(args.hidden, len(word_dict), len(field_dict)).to(device)
        add_optimizer(src_generator, (src2src_optimizers, trg2src_optimizers))
        add_optimizer(trg_generator, (trg2trg_optimizers, src2trg_optimizers))

    logger.debug('src generator is running on cuda: %d', next(src_generator.parameters()).is_cuda)
    logger.debug('trg generator is running on cuda: %d', next(src_generator.parameters()).is_cuda)

    # Build encoder
    src_enc = RNNEncoder(word_embedding_size=word_embedding_size, field_embedding_size=field_embedding_size,
                         hidden_size=args.hidden, bidirectional=not args.disable_bidirectional,
                         layers=args.layers, dropout=args.dropout).to(device)

    if args.shared_enc:
        trg_enc = src_enc
        add_optimizer(src_enc, (src2src_optimizers, src2trg_optimizers, trg2trg_optimizers, trg2src_optimizers))
    else:
        trg_enc = RNNEncoder(word_embedding_size=word_embedding_size, field_embedding_size=field_embedding_size,
                             hidden_size=args.hidden, bidirectional=not args.disable_bidirectional,
                             layers=args.layers, dropout=args.dropout).to(device)
        add_optimizer(src_enc, (src2src_optimizers, src2trg_optimizers))
        add_optimizer(trg_enc, (trg2trg_optimizers, trg2src_optimizers))

    logger.debug('encoder model is running on cuda: %d', next(src_enc.parameters()).is_cuda)

    # Build decoders
    src_dec = RNNAttentionDecoder(word_embedding_size=word_embedding_size,
                                  field_embedding_size=field_embedding_size, hidden_size=args.hidden,
                                  layers=args.layers, dropout=args.dropout, input_feeding=False).to(device)

    if args.shared_dec:
        trg_dec = src_dec
        add_optimizer(src_dec, (src2src_optimizers, trg2src_optimizers, trg2trg_optimizers, src2trg_optimizers))
    else:
        trg_dec = RNNAttentionDecoder(word_embedding_size=word_embedding_size,
                                      field_embedding_size=field_embedding_size, hidden_size=args.hidden,
                                      layers=args.layers, dropout=args.dropout, input_feeding=False).to(device)
        add_optimizer(src_dec, (src2src_optimizers, trg2src_optimizers))
        add_optimizer(trg_dec, (trg2trg_optimizers, src2trg_optimizers))

    logger.debug('decoder model is running on cuda: %d', next(src_dec.parameters()).is_cuda)
    logger.debug('attention model is running on cuda: %d', next(src_dec.attention.parameters()).is_cuda)

    discriminator = None

    if (args.corpus_mode == 'mono') and not args.disable_discriminator:
        discriminator = Discriminator(args.hidden, args.dis_hidden, args.n_dis_layers, args.dropout)
        discriminator = discriminator.to(device)

    # Build translators
    src2src_translator = Translator("src2src",
                                    encoder_word_embeddings=src_encoder_word_embeddings,
                                    decoder_word_embeddings=src_decoder_word_embeddings,
                                    encoder_field_embeddings=src_encoder_field_embeddings,
                                    decoder_field_embeddings=src_decoder_field_embeddings,
                                    generator=src_generator,
                                    src_word_dict=word_dict, trg_word_dict=word_dict,
                                    src_field_dict=field_dict, trg_field_dict=field_dict,
                                    src_type=src_type, trg_type=src_type, w_sos_id=w_sos_id[src_type],
                                    bpemb_en=bpemb_en, encoder=src_enc, decoder=src_dec, discriminator=discriminator,
                                    denoising=args.denoising_mode, device=device,
                                    max_word_shuffle_distance=args.word_shuffle,
                                    word_dropout_prob=args.word_dropout,
                                    word_blanking_prob=args.word_blank)
    src2trg_translator = Translator("src2trg",
                                    encoder_word_embeddings=src_encoder_word_embeddings,
                                    decoder_word_embeddings=trg_decoder_word_embeddings,
                                    encoder_field_embeddings=src_encoder_field_embeddings,
                                    decoder_field_embeddings=trg_decoder_field_embeddings,
                                    generator=trg_generator,
                                    src_word_dict=word_dict, trg_word_dict=word_dict,
                                    src_field_dict=field_dict, trg_field_dict=field_dict,
                                    src_type=src_type, trg_type=trg_type, w_sos_id=w_sos_id[trg_type],
                                    bpemb_en=bpemb_en, encoder=src_enc, decoder=trg_dec, discriminator=discriminator,
                                    denoising=0, device=device,
                                    max_word_shuffle_distance=args.word_shuffle,
                                    word_dropout_prob=args.word_dropout,
                                    word_blanking_prob=args.word_blank)
    trg2trg_translator = Translator("trg2trg",
                                    encoder_word_embeddings=trg_encoder_word_embeddings,
                                    decoder_word_embeddings=trg_decoder_word_embeddings,
                                    encoder_field_embeddings=trg_encoder_field_embeddings,
                                    decoder_field_embeddings=trg_decoder_field_embeddings,
                                    generator=trg_generator,
                                    src_word_dict=word_dict, trg_word_dict=word_dict,
                                    src_field_dict=field_dict, trg_field_dict=field_dict,
                                    src_type=trg_type, trg_type=trg_type, w_sos_id=w_sos_id[trg_type],
                                    bpemb_en=bpemb_en, encoder=trg_enc, decoder=trg_dec, discriminator=discriminator,
                                    denoising=args.denoising_mode, device=device,
                                    max_word_shuffle_distance=args.word_shuffle,
                                    word_dropout_prob=args.word_dropout,
                                    word_blanking_prob=args.word_blank)
    trg2src_translator = Translator("trg2src",
                                    encoder_word_embeddings=trg_encoder_word_embeddings,
                                    decoder_word_embeddings=src_decoder_word_embeddings,
                                    encoder_field_embeddings=trg_encoder_field_embeddings,
                                    decoder_field_embeddings=src_decoder_field_embeddings,
                                    generator=src_generator,
                                    src_word_dict=word_dict, trg_word_dict=word_dict,
                                    src_field_dict=field_dict, trg_field_dict=field_dict,
                                    src_type=trg_type, trg_type=src_type, w_sos_id=w_sos_id[src_type],
                                    bpemb_en=bpemb_en, encoder=trg_enc, decoder=src_dec, discriminator=discriminator,
                                    denoising=0, device=device,
                                    max_word_shuffle_distance=args.word_shuffle,
                                    word_dropout_prob=args.word_dropout,
                                    word_blanking_prob=args.word_blank)

    # Build trainers
    trainers = []
    iters_per_epoch = int(np.ceil(corpus_size / args.batch))
    print("CORPUS_SIZE = %d | BATCH_SIZE = %d | ITERS_PER_EPOCH = %d" % (corpus_size, args.batch, iters_per_epoch))

    if args.corpus_mode == 'mono':
        f_content = open(src_corpus_path + '.content', encoding=args.encoding, errors='surrogateescape')
        f_labels = open(src_corpus_path + '.labels', encoding=args.encoding, errors='surrogateescape')
        src_corpus_path = data.CorpusReader(f_content, f_labels, max_sentence_length=args.max_sentence_length,
                                       cache_size=args.cache)
        f_content = open(trg_corpus_path + '.content', encoding=args.encoding, errors='surrogateescape')
        f_labels = open(trg_corpus_path + '.labels', encoding=args.encoding, errors='surrogateescape')
        trg_corpus_path = data.CorpusReader(f_content, f_labels, max_sentence_length=args.max_sentence_length,
                                       cache_size=args.cache)

        if not args.disable_discriminator:
            disc_trainer = DiscTrainer(device, src_corpus_path, trg_corpus_path, src_enc, trg_enc, src_encoder_word_embeddings,
                                       src_encoder_field_embeddings, word_dict, field_dict, discriminator,
                                       args.learning_rate, batch_size=args.batch)
            trainers.append(disc_trainer)

        src2src_trainer = Trainer(translator=src2src_translator, optimizers=src2src_optimizers, corpus=src_corpus_path,
                                  batch_size=args.batch, iters_per_epoch=iters_per_epoch)
        trainers.append(src2src_trainer)
        if not args.disable_backtranslation:
            trgback2src_trainer = Trainer(translator=trg2src_translator, optimizers=trg2src_optimizers,
                                          corpus=data.BacktranslatorCorpusReader(corpus=src_corpus_path,
                                                                                 translator=src2trg_translator),
                                          batch_size=args.batch, iters_per_epoch=iters_per_epoch)
            trainers.append(trgback2src_trainer)

        trg2trg_trainer = Trainer(translator=trg2trg_translator, optimizers=trg2trg_optimizers, corpus=trg_corpus_path,
                                  batch_size=args.batch, iters_per_epoch=iters_per_epoch)
        trainers.append(trg2trg_trainer)
        if not args.disable_backtranslation:
            srcback2trg_trainer = Trainer(translator=src2trg_translator, optimizers=src2trg_optimizers,
                                          corpus=data.BacktranslatorCorpusReader(corpus=trg_corpus_path,
                                                                                 translator=trg2src_translator),
                                          batch_size=args.batch, iters_per_epoch=iters_per_epoch)
            trainers.append(srcback2trg_trainer)
    elif args.corpus_mode == 'para':
        fsrc_content = open(src_corpus_path + '.content', encoding=args.encoding, errors='surrogateescape')
        fsrc_labels = open(src_corpus_path + '.labels', encoding=args.encoding, errors='surrogateescape')
        ftrg_content = open(trg_corpus_path + '.content', encoding=args.encoding, errors='surrogateescape')
        ftrg_labels = open(trg_corpus_path + '.labels', encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(fsrc_content, fsrc_labels, trg_word_file=ftrg_content, trg_field_file=ftrg_labels,
                                   max_sentence_length=args.max_sentence_length,
                                   cache_size=args.cache)
        src2trg_trainer = Trainer(translator=src2trg_translator, optimizers=src2trg_optimizers, corpus=corpus,
                                  batch_size=args.batch, iters_per_epoch=iters_per_epoch)
        trainers.append(src2trg_trainer)

    # Build validators
    if args.src_valid_corpus != '' and args.trg_valid_corpus != '':
        with ExitStack() as stack:
            src_content_vfile = stack.enter_context(open(args.src_valid_corpus + '.content', encoding=args.encoding,
                                                         errors='surrogateescape'))
            src_labels_vfile = stack.enter_context(open(args.src_valid_corpus + '.labels', encoding=args.encoding,
                                                        errors='surrogateescape'))
            trg_content_vfile = stack.enter_context(open(args.trg_valid_corpus + '.content', encoding=args.encoding,
                                                         errors='surrogateescape'))
            trg_labels_vfile = stack.enter_context(open(args.trg_valid_corpus + '.labels', encoding=args.encoding,
                                                        errors='surrogateescape'))

            src_content = src_content_vfile.readlines()
            src_labels = src_labels_vfile.readlines()
            trg_content = trg_content_vfile.readlines()
            trg_labels = trg_labels_vfile.readlines()
            assert len(src_content) == len(trg_content) == len(src_labels) == len(trg_labels), \
                "Validation sizes do not match {} {} {} {}".format(len(src_content), len(trg_content), len(src_labels),
                len(trg_labels))

            src_content = [list(map(int, line.strip().split())) for line in src_content]
            src_labels = [list(map(int, line.strip().split())) for line in src_labels]
            trg_content = [list(map(int, line.strip().split())) for line in trg_content]
            trg_labels = [list(map(int, line.strip().split())) for line in trg_labels]

            cache = []
            for src_sent, src_label, trg_sent, trg_label in zip(src_content, src_labels, trg_content, trg_labels):
                if 0 < len(src_sent) <= args.max_sentence_length and 0 < len(trg_sent) <= args.max_sentence_length:
                    cache.append((src_sent, src_label, trg_sent, trg_label))

            src_content, src_labels, trg_content, trg_labels = zip(*cache)

            src2trg_validator = Validator(src2trg_translator, src_content, trg_content, src_labels, trg_labels)

            if args.corpus_mode == 'mono':
                src2src_validator = Validator(src2src_translator, src_content, src_content, src_labels, src_labels)

                trg2src_validator = Validator(trg2src_translator, trg_content, src_content, trg_labels, src_labels)

                trg2trg_validator = Validator(trg2trg_translator, trg_content, trg_content, trg_labels, trg_labels)

            del src_content
            del src_labels
            del trg_content
            del trg_labels
    else:
        src2src_validator = None
        src2trg_validator = None
        trg2src_validator = None
        trg2trg_validator = None

    # Build loggers
    loggers = []
    semi_loggers = []

    if args.corpus_mode == 'mono':
        if not args.disable_backtranslation:
            loggers.append(Logger('Source to target (backtranslation)', srcback2trg_trainer, src2trg_validator,
                                  None, args.encoding, short_name='src2trg_bt', train_writer=train_writer,
                                  valid_writer=valid_writer))
            loggers.append(Logger('Target to source (backtranslation)', trgback2src_trainer, trg2src_validator,
                                  None, args.encoding, short_name='trg2src_bt', train_writer=train_writer,
                                  valid_writer=valid_writer))

        loggers.append(Logger('Source to source', src2src_trainer, src2src_validator, None, args.encoding,
                              short_name='src2src', train_writer=train_writer, valid_writer=valid_writer))
        loggers.append(Logger('Target to target', trg2trg_trainer, trg2trg_validator, None, args.encoding,
                              short_name='trg2trg', train_writer=train_writer, valid_writer=valid_writer))
    elif args.corpus_mode == 'para':
        loggers.append(Logger('Source to target', src2trg_trainer, src2trg_validator, None, args.encoding,
                              short_name='src2trg_para', train_writer=train_writer, valid_writer=valid_writer))

    # Method to save models
    def save_models(name):
        # torch.save(src2src_translator, '{0}.{1}.src2src.pth'.format(args.save, name))
        # torch.save(trg2trg_translator, '{0}.{1}.trg2trg.pth'.format(args.save, name))
        torch.save(src2trg_translator, '{0}.{1}.src2trg.pth'.format(args.save, name))
        if args.corpus_mode == 'mono':
            torch.save(trg2src_translator, '{0}.{1}.trg2src.pth'.format(args.save, name))

    ref_string_path = args.trg_valid_corpus + '.str.content'

    if not os.path.isfile(ref_string_path):
        print("Creating ref file... [%s]" % (ref_string_path))

        with ExitStack() as stack:

            fref_content = stack.enter_context(
                open(args.trg_valid_corpus + '.content', encoding=args.encoding, errors='surrogateescape'))
            fref_str_content = stack.enter_context(
                open(ref_string_path, mode='w', encoding=args.encoding, errors='surrogateescape'))

            for line in fref_content:
                ref_ids = [int(idstr) for idstr in line.strip().split()]
                ref_str = bpemb_en.decode_ids(ref_ids)
                fref_str_content.write(ref_str + '\n')

        print("Ref file created!")

    # Training
    for curr_iter in range(1, args.iterations + 1):
        print_dbg = (0 != args.dbg_print_interval) and (curr_iter % args.dbg_print_interval == 0)

        for trainer in trainers:
            trainer.step(print_dbg=print_dbg, include_field_loss=not args.disable_field_loss)

        if args.save is not None and args.save_interval > 0 and curr_iter % args.save_interval == 0:
            save_models('it{0}'.format(curr_iter))

        if curr_iter % args.log_interval == 0:
            print()
            print('[{0}] TRAIN-STEP {1} x {2}'.format(args.save, curr_iter, args.batch))
            for logger in loggers:
                logger.log(curr_iter)

        if curr_iter % iters_per_epoch == 0:
            save_models('it{0}'.format(curr_iter))
            print()
            print('[{0}] VALID-STEP {1}'.format(args.save, curr_iter))
            for logger in loggers:
                if logger.validator is not None:
                    logger.validate(curr_iter)

            model = '{0}.{1}.src2trg.pth'.format(args.save, 'it{0}'.format(curr_iter))

            bleu_thread = threading.Thread(target=calc_bleu,
                                           args=(model, args.save, args.src_valid_corpus, args.trg_valid_corpus + '.str.result',
                                                 ref_string_path, bpemb_en, curr_iter, args.bleu_device, valid_writer))
            bleu_thread.start()
            if args.cuda == args.bleu_device or args.bleu_device == 'cpu':
                bleu_thread.join()

    save_models('final')
    train_writer.close()
    valid_writer.close()


class DiscTrainer:
    def __init__(self, device, table_corpus, text_corpus, src_enc, trg_enc, encoder_word_embeddings,
                 encoder_field_embeddings, word_dict, field_dict, discriminator, lr, batch_size=50, batch_first=False):
        self.device = device
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.corpus = [table_corpus, text_corpus]
        self.w_eos_id = word_dict.eos_index
        self.w_pad_id = word_dict.pad_index
        self.f_pad_id = field_dict.pad_index
        self.f_null_id = field_dict.null_index
        self.encoder = [src_enc, trg_enc]
        self.encoder_word_embeddings = encoder_word_embeddings
        self.encoder_field_embeddings = encoder_field_embeddings
        self.discriminator = discriminator
        self.optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

    def add_control_sym(self, sentences, sentences_field):
        def valid_and_return(sent, sent_field):
            assert len(sent) == len(sent_field)
            return sent, sent_field, len(sent)

        sent_field_len = [valid_and_return(sent + [self.w_eos_id], sent_field + [self.f_null_id])
                          for sent, sent_field in zip(sentences, sentences_field)]

        sents, sents_field, lengths = zip(*sent_field_len)

        return list(sents), list(sents_field), list(lengths)

    def add_padding(self, sents, sents_field, max_length):
        sents = [s + [self.w_pad_id]*(max_length-len(s)) for s in sents]
        sents_field = [s + [self.f_pad_id] * (max_length - len(s)) for s in sents_field]

        return sents, sents_field

    def preprocess_ids(self, sentences, sentences_field):
        word_ids, field_ids, lengths = self.add_control_sym(sentences, sentences_field)
        max_length = max(lengths)

        # Padding
        word_ids, field_ids = self.add_padding(word_ids, field_ids, max_length)

        return word_ids, field_ids, lengths

    def encode(self, encoder, sentences, sentences_field):
        word_ids, field_ids, lengths = self.preprocess_ids(sentences, sentences_field)

        with torch.no_grad():
            if not self.batch_first:
                var_wordids = torch.LongTensor(word_ids).transpose(1, 0).to(self.device)
                var_fieldids = torch.LongTensor(field_ids).transpose(1, 0).to(self.device)
                var_length = torch.LongTensor(lengths).to(self.device)
            else:
                var_wordids = torch.LongTensor(word_ids).to(self.device)
                var_fieldids = torch.LongTensor(field_ids).to(self.device)
                var_length = torch.LongTensor(lengths).to(self.device)

        hidden = encoder.initial_hidden(len(sentences)).to(self.device)

        hidden, context = encoder(word_ids=var_wordids, field_ids=var_fieldids, lengths=var_length,
                                  word_embeddings=self.encoder_word_embeddings,
                                  field_embeddings=self.encoder_field_embeddings, hidden=hidden)
        return hidden, context, var_length

    def step(self, print_dbg=False, include_field_loss=False):
        self.encoder_word_embeddings.eval()
        self.encoder_field_embeddings.eval()
        self.discriminator.train()

        self.optimizer.zero_grad()

        # Read input sentences
        encoded = []
        t = time.time()
        for corpus_id, corpus in enumerate(self.corpus):
            src_word, trg_word, src_field, trg_field = corpus.next_batch(self.batch_size)
            with torch.no_grad():
                self.encoder[corpus_id].eval()
                hidden, context, context_lengths = self.encode(self.encoder[corpus_id], src_word, src_field)
                encoded.append(hidden)

        # discriminator
        dis_inputs = [x.view(-1, x.size(-1)) for x in encoded]
        ntokens = [dis_input.size(0) for dis_input in dis_inputs]
        encoded = torch.cat(dis_inputs, 0)
        predictions = self.discriminator(encoded.data)

        # loss
        dis_target = torch.cat([torch.zeros(sz).fill_(i) for i, sz in enumerate(ntokens)])
        dis_target = dis_target.contiguous().long().to(self.device)
        y = dis_target

        loss = F.cross_entropy(predictions, y)

        if print_dbg:
            print("Discriminator training loss %f" % (loss.item()))

        # optimizer
        loss.backward()
        self.optimizer.step()


class Trainer:
    def __init__(self, corpus, optimizers, translator, batch_size=50, iters_per_epoch=1):
        self.corpus = corpus
        self.translator = translator
        self.optimizers = optimizers
        self.batch_size = batch_size
        self.iters_per_epoch = iters_per_epoch
        self.reset_stats()

    def step(self, print_dbg=False, include_field_loss=True):
        # Reset gradients
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # Read input sentences
        t = time.time()
        src_word, trg_word, src_field, trg_field = self.corpus.next_batch(self.batch_size)
        self.src_sent_batch_count += 1
        self.src_word_count += sum([len(sentence) + 1 for sentence in src_word])  # TODO Depends on special symbols EOS/SOS
        self.trg_word_count += sum([len(sentence) + 1 for sentence in trg_word])  # TODO Depends on special symbols EOS/SOS
        self.io_time += time.time() - t

        # Compute loss
        t = time.time()
        word_loss, field_loss, dis_loss = self.translator.score(src_word, trg_word, src_field, trg_field,
                                                               print_dbg=print_dbg, train=True)

        self.word_loss += word_loss.item()
        self.field_loss += field_loss.item()
        self.dis_loss += dis_loss.item()

        total_loss = word_loss

        if include_field_loss:
            total_loss += field_loss

        self.forward_time += time.time() - t

        # Backpropagate error + optimize
        t = time.time()

        total_loss = total_loss.div(self.batch_size)
        total_loss += dis_loss

        total_loss.backward()
        for optimizer in self.optimizers:
            optimizer.step()
        self.backward_time += time.time() - t

    def reset_stats(self):
        self.src_sent_batch_count = 0
        self.src_word_count = 0
        self.trg_word_count = 0
        self.io_time = 0
        self.forward_time = 0
        self.backward_time = 0
        self.word_loss = 0
        self.field_loss = 0
        self.dis_loss = 0

    def perplexity_per_word(self):
        return np.exp2(self.word_loss/self.trg_word_count/np.log(2))

    def total_time(self):
        return self.io_time + self.forward_time + self.backward_time

    def words_per_second(self):
        return self.src_word_count / self.total_time(),  self.trg_word_count / self.total_time()


class Validator:
    def __init__(self, translator, src_content, ref_content, src_labels, ref_labels, batch_size=50):
        self.translator = translator
        self.batch_size = batch_size
        self.sentence_count = len(src_content)

        # Sorting
        lengths = [len(sentence) for sentence in src_content]
        true2sorted = sorted(range(self.sentence_count), key=lambda x: -lengths[x])
        self.src_content = [src_content[i] for i in true2sorted]
        self.ref_content = [ref_content[i] for i in true2sorted]
        self.src_labels = [src_labels[i] for i in true2sorted]
        self.ref_labels = [ref_labels[i] for i in true2sorted]

        self.ref_word_cnt = sum(
            [len(sentence) + 1 for sentence in ref_content])  # TODO Depends on special symbols EOS/SOS

    def evaluate(self):
        w_loss = 0
        f_loss = 0
        d_loss = 0
        batch_cnt = np.ceil(self.sentence_count / self.batch_size)
        with torch.no_grad():
            for i in range(0, self.sentence_count, self.batch_size):
                j = min(i + self.batch_size, self.sentence_count)

                word_loss, field_loss, diss_loss = self.translator.score(
                    self.src_content[i:j],
                    self.ref_content[i:j],
                    self.src_labels[i:j],
                    self.ref_labels[i:j])

                w_loss += word_loss.item()
                f_loss += field_loss.item()
                d_loss += diss_loss.item()

        return w_loss / self.ref_word_cnt, f_loss / self.ref_word_cnt, d_loss / batch_cnt

    @staticmethod
    def perplexity_per_word(avg_word_loss):
        return np.exp2(avg_word_loss/np.log(2))


class Logger:
    def __init__(self, full_name, trainer, validator=None, output_prefix=None, encoding='utf-8', short_name=None,
                 train_writer=None, valid_writer=None):
        self.full_name = full_name
        self.short_name = short_name
        self.trainer = trainer
        self.validator = validator
        self.train_writer = train_writer
        self.valid_writer = valid_writer
        self.output_prefix = output_prefix
        self.encoding = encoding

    def log(self, step=0):
        w_loss = self.trainer.word_loss / self.trainer.trg_word_count
        f_loss = self.trainer.field_loss / self.trainer.trg_word_count
        dis_loss = self.trainer.dis_loss / self.trainer.src_sent_batch_count
        ppl = self.trainer.perplexity_per_word()
        epoch = int(np.ceil(step / self.trainer.iters_per_epoch))

        print('{0}'.format(self.full_name))
        print('  - Training:     ppl {0:6.3f} | w_loss {1:3.4f} | f_loss {2:3.4f} | dis_loss {3:3.4f}'
              '  (t_time {4:.2f}s io_time {5:.2f}s; fw_time {6:.2f}s; bw_time {7:.2f}s: '
              '{8:.2f}tok/s src, {9:.2f}tok/s trg; epoch {10})'
              .format(ppl, w_loss, f_loss, dis_loss, self.trainer.total_time(),
                      self.trainer.io_time, self.trainer.forward_time, self.trainer.backward_time,
                      self.trainer.words_per_second()[0], self.trainer.words_per_second()[1],
                      epoch))
        self.trainer.reset_stats()

        if self.train_writer is not None:
            self.train_writer.add_scalar(self.short_name + '/word_loss', w_loss, step)
            self.train_writer.add_scalar(self.short_name + '/field_loss', f_loss, step)
            self.train_writer.add_scalar(self.short_name + '/disc_loss', dis_loss, step)
            self.train_writer.add_scalar(self.short_name + '/ppl', ppl, step)

        sys.stdout.flush()

    def validate(self, step):
        t = time.time()
        avg_w_loss, avg_f_loss, avg_diss_loss = self.validator.evaluate()
        ppl = self.validator.perplexity_per_word(avg_w_loss)
        t = time.time() - t
        print('{0}'.format(self.full_name))
        print('  - Validation: ppl {0:6.3f} | w_loss {1:3.4f} | f_loss {2:3.4f} | dis_loss {3:3.4f} | time {4:.2f}s'
              .format(ppl, avg_w_loss, avg_f_loss, avg_diss_loss, t))
        if self.valid_writer is not None:
            self.valid_writer.add_scalar(self.short_name + '/word_loss', avg_w_loss, step)
            self.valid_writer.add_scalar(self.short_name + '/field_loss', avg_f_loss, step)
            self.valid_writer.add_scalar(self.short_name + '/disc_loss', avg_diss_loss, step)
            self.valid_writer.add_scalar(self.short_name + '/ppl', ppl, step)
