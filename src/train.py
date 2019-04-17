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


def main_train():
    # Build argument parser
    parser = argparse.ArgumentParser(description='Train a table to text model')

    # Training corpus
    corpora_group = parser.add_argument_group('training corpora', 'Corpora related arguments; specify either unaligned or aligned training corpora (or both)')
    # "Languages (type,path)"
    corpora_group.add_argument('--src_corpus_params', type=str, default='table, ./data/processed_data/train/train.box',
                               help='the source unaligned corpus (type,path). Type = text/table')
    corpora_group.add_argument('--trg_corpus_params', type=str, default='text, ./data/processed_data/train/train.article',
                               help='the target unaligned corpus (type,path). Type = text/table')
    # Maybe add src/target type (i.e. text/table)
    corpora_group.add_argument('--corpus_mode', type=str, default='mono',
                               help='training mode: "mono" (unsupervised) / "para" (supervised)')

    corpora_group.add_argument('--max_sentence_length', type=int, default=50, help='the maximum sentence length for training (defaults to 50)')
    corpora_group.add_argument('--cache', type=int, default=100000, help='the cache size (in sentences) for corpus reading (defaults to 1000000)')
    corpora_group.add_argument('--cache_parallel', type=int, default=None, help='the cache size (in sentences) for parallel corpus reading')

    # Embeddings/vocabulary
    embedding_group = parser.add_argument_group('embeddings', 'Embedding related arguments; either give pre-trained embeddings, or a vocabulary and embedding dimensionality to randomly initialize them')
    embedding_group.add_argument('--emb_dim', type=int, default=100, help='the number of dimensions for the embedding layer')
    embedding_group.add_argument('--word_vocab_size', type=int, default=100, help='word vocabulary size')
    embedding_group.add_argument('--cutoff', type=int, default=0, help='cutoff vocabulary to the given size')

    # ???
    embedding_group.add_argument('--learn_encoder_embeddings', action='store_true', help='learn the encoder embeddings instead of using the pre-trained ones')
    embedding_group.add_argument('--fixed_decoder_embeddings', action='store_true', help='use fixed embeddings in the decoder instead of learning them from scratch')
    embedding_group.add_argument('--fixed_generator', action='store_true', help='use fixed embeddings in the output softmax instead of learning it from scratch')

    # Architecture
    architecture_group = parser.add_argument_group('architecture', 'Architecture related arguments')
    architecture_group.add_argument('--layers', type=int, default=2, help='the number of encoder/decoder layers (defaults to 2)')
    architecture_group.add_argument('--hidden', type=int, default=600, help='the number of dimensions for the hidden layer (defaults to 600)')
    architecture_group.add_argument('--dis_hidden', type=int, default=150, help='Number of dimensions for the discriminator hidden layers')
    architecture_group.add_argument('--n_dis_layers', type=int, default=2, help='Number of discriminator layers')
    architecture_group.add_argument('--disable_bidirectional', action='store_true', help='use a single direction encoder')
    architecture_group.add_argument('--disable_denoising', action='store_true', help='disable random swaps')
    architecture_group.add_argument('--disable_backtranslation', action='store_true', help='disable backtranslation')
    architecture_group.add_argument('--disable_field_loss', action='store_true', help='disable backtranslation')
    architecture_group.add_argument('--disable_discriminator', action='store_true', help='disable discriminator')
    architecture_group.add_argument('--shared_enc', action='store_true', help='share enc for both directions')
    architecture_group.add_argument('--shared_dec', action='store_true', help='share dec for both directions')


    # Optimization
    optimization_group = parser.add_argument_group('optimization', 'Optimization related arguments')
    optimization_group.add_argument('--batch', type=int, default=50, help='the batch size (defaults to 50)')
    optimization_group.add_argument('--learning_rate', type=float, default=0.0002, help='the global learning rate (defaults to 0.0002)')
    optimization_group.add_argument('--dropout', metavar='PROB', type=float, default=0.3, help='dropout probability for the encoder/decoder (defaults to 0.3)')
    optimization_group.add_argument('--param_init', metavar='RANGE', type=float, default=0.1, help='uniform initialization in the specified range (defaults to 0.1,  0 for module specific default initialization)')
    optimization_group.add_argument('--iterations', type=int, default=300000, help='the number of training iterations (defaults to 300000)')

    # Model saving
    saving_group = parser.add_argument_group('model saving', 'Arguments for saving the trained model')
    saving_group.add_argument('--save', metavar='PREFIX', help='save models with the given prefix')
    saving_group.add_argument('--save_interval', type=int, default=0, help='save intermediate models at this interval')

    # Logging/validation
    logging_group = parser.add_argument_group('logging', 'Logging and validation arguments')
    logging_group.add_argument('--log_interval', type=int, default=1000, help='log at this interval (defaults to 1000)')
    logging_group.add_argument('--src_valid_corpus', type=str, default='./data/processed_data/valid/valid.box')
    logging_group.add_argument('--trg_valid_corpus', type=str, default='./data/processed_data/valid/valid.article')
    logging_group.add_argument('--validation', nargs='+', default=(), help='use parallel corpora for validation')
    logging_group.add_argument('--validation_directions', nargs='+', default=['src2src', 'trg2trg', 'src2trg', 'trg2src'], help='validation directions')
    logging_group.add_argument('--validation_output', metavar='PREFIX', help='output validation translations with the given prefix')
    logging_group.add_argument('--validation_beam_size', type=int, default=0, help='use beam search for validation')
    logging_group.add_argument('--print_level', type=str, default='info', help='logging level [debug | info]')

    # Other
    parser.add_argument('--preprocess_metadata_path', type=str, default='', help='Path of bin file containing preprocess metadata')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--cuda', type=str, default='cuda:0')

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

    print("Log every %d intervals" % args.log_interval)

    # Validate arguments
    if args.src_corpus_params is None or args.trg_corpus_params is None:
        print("Must supply corpus")
        sys.exit(-1)

    args.src_corpus_params = args.src_corpus_params.split(',')
    args.trg_corpus_params = args.trg_corpus_params.split(',')
    assert len(args.src_corpus_params) == 2
    assert len(args.trg_corpus_params) == 2

    src_type, src_corpus = args.src_corpus_params
    trg_type, trg_corpus = args.trg_corpus_params

    src_type = src_type.strip()
    src_corpus = src_corpus.strip()
    trg_type = trg_type.strip()
    trg_corpus = trg_corpus.strip()

    assert src_type != trg_type
    assert (src_type in ['table', 'text']) and (trg_type in ['table', 'text'])

    # Select device
    if torch.cuda.is_available():
        device = torch.device(args.cuda)
    else:
        device = torch.device('cpu')

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

    if os.path.isfile(args.preprocess_metadata_path):
        metadata = torch.load(args.preprocess_metadata_path)
        bpemb_en = metadata.init_bpe_module()
        word_dict: BpeWordDict = torch.load(metadata.word_dict_path)
        field_dict: LabelDict = torch.load(metadata.field_dict_path)
    else:
        bpemb_en, word_dict, field_dict = preprocess(args.emb_dim, args.word_vocab_size)

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

    if not args.disable_discriminator:
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
                                    denoising=not args.disable_denoising, device=device)
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
                                    denoising=False, device=device)
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
                                    denoising=not args.disable_denoising, device=device)
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
                                    denoising=False, device=device)

    # Build trainers
    trainers = []

    if args.corpus_mode == 'mono':
        if args.src_corpus_params is not None:
            f_content = open(src_corpus + '.content', encoding=args.encoding, errors='surrogateescape')
            f_labels = open(src_corpus + '.labels', encoding=args.encoding, errors='surrogateescape')
            src_corpus = data.CorpusReader(f_content, f_labels, max_sentence_length=args.max_sentence_length,
                                           cache_size=args.cache)

        if args.trg_corpus_params is not None:
            f_content = open(trg_corpus + '.content', encoding=args.encoding, errors='surrogateescape')
            f_labels = open(trg_corpus + '.labels', encoding=args.encoding, errors='surrogateescape')
            trg_corpus = data.CorpusReader(f_content, f_labels, max_sentence_length=args.max_sentence_length,
                                           cache_size=args.cache)

        if not args.disable_discriminator:
            disc_trainer = DiscTrainer(device, src_corpus, trg_corpus, src_enc, trg_enc, src_encoder_word_embeddings,
                                       src_encoder_field_embeddings, word_dict, field_dict, discriminator,
                                       args.learning_rate, batch_size=args.batch)
            trainers.append(disc_trainer)

        if args.src_corpus_params is not None:
            src2src_trainer = Trainer(translator=src2src_translator, optimizers=src2src_optimizers, corpus=src_corpus,
                                      batch_size=args.batch)
            trainers.append(src2src_trainer)
            if not args.disable_backtranslation:
                trgback2src_trainer = Trainer(translator=trg2src_translator, optimizers=trg2src_optimizers,
                                              corpus=data.BacktranslatorCorpusReader(corpus=src_corpus,
                                                                                     translator=src2trg_translator),
                                              batch_size=args.batch)
                trainers.append(trgback2src_trainer)
        if args.trg_corpus_params is not None:
            trg2trg_trainer = Trainer(translator=trg2trg_translator, optimizers=trg2trg_optimizers, corpus=trg_corpus,
                                      batch_size=args.batch)
            trainers.append(trg2trg_trainer)
            if not args.disable_backtranslation:
                srcback2trg_trainer = Trainer(translator=src2trg_translator, optimizers=src2trg_optimizers,
                                              corpus=data.BacktranslatorCorpusReader(corpus=trg_corpus,
                                                                                     translator=trg2src_translator),
                                              batch_size=args.batch)
                trainers.append(srcback2trg_trainer)
    elif args.corpus_mode == 'para':
        fsrc_content = open(src_corpus + '.content', encoding=args.encoding, errors='surrogateescape')
        fsrc_labels = open(src_corpus + '.labels', encoding=args.encoding, errors='surrogateescape')
        ftrg_content = open(trg_corpus + '.content', encoding=args.encoding, errors='surrogateescape')
        ftrg_labels = open(trg_corpus + '.labels', encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(fsrc_content, fsrc_labels, trg_word_file=ftrg_content, trg_field_file=ftrg_labels,
                                   max_sentence_length=args.max_sentence_length,
                                   cache_size=args.cache if args.cache_parallel is None else args.cache_parallel)
        src2trg_trainer = Trainer(translator=src2trg_translator, optimizers=src2trg_optimizers, corpus=corpus,
                                  batch_size=args.batch)
        trainers.append(src2trg_trainer)

    # Build validators
    src2trg_validators = []
    trg2src_validators = []

    if 0:
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
            if len(src_content) != len(trg_content) != len(src_labels) != len(trg_labels):
                print('Validation sizes do not match')
                sys.exit(-1)

            src_content = list(map(lambda x: list(map(lambda y: int(y), x.strip().split())), src_content))
            src_labels = list(map(lambda x: list(map(lambda y: int(y), x.strip().split())), src_labels))
            trg_content = list(map(lambda x: list(map(lambda y: int(y), x.strip().split())), trg_content))
            trg_labels = list(map(lambda x: list(map(lambda y: int(y), x.strip().split())), trg_labels))

            src2trg_validators.append(Validator(src2trg_translator, src_content, trg_content,
                                                src_labels, trg_labels, args.batch, 0))
            trg2src_validators.append(Validator(trg2src_translator, trg_content, src_content,
                                                trg_labels, src_labels, args.batch, 0))

    # Build loggers
    loggers = []

    if args.corpus_mode == 'mono':
        loggers.append(Logger('Source to target (backtranslation)', srcback2trg_trainer, [], None,
                              args.encoding))
        loggers.append(Logger('Target to source (backtranslation)', trgback2src_trainer, [], None,
                              args.encoding))
        loggers.append(Logger('Source to source', src2src_trainer, [], None, args.encoding))
        loggers.append(Logger('Target to target', trg2trg_trainer, [], None, args.encoding))
    elif args.corpus_mode == 'para':
        loggers.append(Logger('Source to target', src2trg_trainer, src2trg_validators, None, args.encoding))

    # Method to save models
    def save_models(name):
        # torch.save(src2src_translator, '{0}.{1}.src2src.pth'.format(args.save, name))
        # torch.save(trg2trg_translator, '{0}.{1}.trg2trg.pth'.format(args.save, name))
        torch.save(src2trg_translator, '{0}.{1}.src2trg.pth'.format(args.save, name))
        if args.corpus_mode == 'mono':
            torch.save(trg2src_translator, '{0}.{1}.trg2src.pth'.format(args.save, name))

    # Training
    for curr_iter in range(1, args.iterations + 1):
        print_dbg = (curr_iter % args.log_interval == 0)

        for trainer in trainers:
            trainer.step(print_dbg=print_dbg, include_field_loss=not args.disable_field_loss)

        if args.save is not None and args.save_interval > 0 and curr_iter % args.save_interval == 0:
            save_models('it{0}'.format(curr_iter))

        if print_dbg:
            print()
            print('STEP {0} x {1}'.format(curr_iter, args.batch))
            for logger in loggers:
                logger.log(curr_iter)

    save_models('final')


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
            else:
                var_wordids = torch.LongTensor(word_ids).to(self.device)
                var_fieldids = torch.LongTensor(field_ids).to(self.device)

            # logger.debug('enc: word_ids are on cuda: %d', var_wordids.is_cuda)
            # logger.debug('enc: field_ids are on cuda: %d', var_fieldids.is_cuda)

        hidden = encoder.initial_hidden(len(sentences)).to(self.device)
        # logger.debug('hidden is on cuda: %d', hidden.is_cuda)

        hidden, context = encoder(word_ids=var_wordids, field_ids=var_fieldids, lengths=lengths,
                                  word_embeddings=self.encoder_word_embeddings,
                                  field_embeddings=self.encoder_field_embeddings, hidden=hidden)
        return hidden, context, lengths

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
    def __init__(self, corpus, optimizers, translator, batch_size=50):
        self.corpus = corpus
        self.translator = translator
        self.optimizers = optimizers
        self.batch_size = batch_size
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
        return np.exp(self.word_loss/self.trg_word_count)

    def total_time(self):
        return self.io_time + self.forward_time + self.backward_time

    def words_per_second(self):
        return self.src_word_count / self.total_time(),  self.trg_word_count / self.total_time()


class Validator:
    def __init__(self, translator, src_content, ref_content, src_labels, ref_labels, batch_size=50, beam_size=0):
        self.translator = translator
        self.src_content = src_content
        self.ref_content = ref_content
        self.src_labels = src_labels
        self.ref_labels = ref_labels
        self.sentence_count = len(src_content)
        self.reference_word_count = sum([len(sentence) + 1 for sentence in self.ref_content])  # TODO Depends on special symbols EOS/SOS
        self.batch_size = batch_size
        self.beam_size = beam_size

        # Sorting
        lengths = [len(sentence) for sentence in self.src_content]
        self.true2sorted = sorted(range(self.sentence_count), key=lambda x: -lengths[x])
        self.sorted2true = sorted(range(self.sentence_count), key=lambda x: self.true2sorted[x])
        self.sorted_src_content = [self.src_content[i] for i in self.true2sorted]
        self.sorted_ref_content = [self.ref_content[i] for i in self.true2sorted]
        self.sorted_src_labels = [self.src_labels[i] for i in self.true2sorted]
        self.sorted_ref_labels = [self.ref_labels[i] for i in self.true2sorted]

    def perplexity(self):
        w_loss = 0
        f_loss = 0
        for i in range(0, self.sentence_count, self.batch_size):
            j = min(i + self.batch_size, self.sentence_count)
            word_loss, field_loss = self.translator.score(
                self.sorted_src_content[i:j],
                self.sorted_ref_content[i:j],
                self.sorted_src_labels[i:j],
                self.sorted_ref_labels[i:j],
                print_dbg=False,
                train=False)
            w_loss += word_loss.item()
            f_loss += field_loss.item()
        return np.exp(w_loss/self.reference_word_count),\
               np.exp(f_loss/self.reference_word_count)

    def translate(self):
        w_translations = []
        f_translations = []
        for i in range(0, self.sentence_count, self.batch_size):
            j = min(i + self.batch_size, self.sentence_count)
            content_batch = self.sorted_src_content[i:j]
            labels_batch = self.sorted_src_labels[i:j]
            if self.beam_size <= 0:
                w_translation, f_translation = self.translator.greedy(content_batch, labels_batch, train=False)
                w_translations.append(w_translation)
                f_translations.append(f_translation)
            else:
                pass
                # translations += self.translator.beam_search(batch, train=False, beam_size=self.beam_size)
        return [w_translations[i] for i in self.sorted2true], [f_translations[i] for i in self.sorted2true]


class Logger:
    def __init__(self, name, trainer, validators=(), output_prefix=None, encoding='utf-8'):
        self.name = name
        self.trainer = trainer
        self.validators = validators
        self.output_prefix = output_prefix
        self.encoding = encoding

    def log(self, step=0):
        if self.trainer is not None or len(self.validators) > 0:
            print('{0}'.format(self.name))
        if self.trainer is not None:
            w_loss = self.trainer.word_loss / self.trainer.trg_word_count
            f_loss = self.trainer.field_loss / self.trainer.trg_word_count
            dis_loss = self.trainer.dis_loss / self.trainer.src_sent_batch_count

            print('  - Training:     pps {0:6.3f} | w_loss {1:3.4f} | f_loss {2:3.4f} | dis_loss {3:3.4f}'
                  '  (t_time {4:.2f}s io_time {5:.2f}s; fw_time {6:.2f}s; bw_time {7:.2f}s: '
                  '{8:.2f}tok/s src, {9:.2f}tok/s trg; epoch {10})'
                  .format(self.trainer.perplexity_per_word(), w_loss, f_loss, dis_loss, self.trainer.total_time(),
                          self.trainer.io_time, self.trainer.forward_time, self.trainer.backward_time,
                          self.trainer.words_per_second()[0], self.trainer.words_per_second()[1], self.trainer.corpus.epoch))
            self.trainer.reset_stats()
        for id, validator in enumerate(self.validators):
            if self.trainer.corpus.validate:
                t = time.time()
                self.trainer.corpus.validate = False
                w_pps, f_pps = validator.perplexity()
                print('  - Validation: w_pps {0:10.3f}; f_pps{1:10.5f}  ({2:.2f}s)'.format(w_pps, f_pps, time.time() - t))
                if self.output_prefix is not None:
                    f = open('{0}.{1}.{2}.txt'.format(self.output_prefix, id, step), mode='w',
                             encoding=self.encoding, errors='surrogateescape')
                    for line in validator.translate():
                        print(line, file=f)
                    f.close()
        sys.stdout.flush()
