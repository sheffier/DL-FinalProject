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

from src import devices
from src.encoder import RNNEncoder
from src.decoder import RNNAttentionDecoder
from src.generator import *
from src.translator import Translator

from src.preprocess import BpeWordDict, LabelDict

import argparse
import numpy as np
import sys
import time

import logging

from src.config import bpemb_en

from torch import nn




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

    corpora_group.add_argument('--max_sentence_length', type=int, default=50, help='the maximum sentence length for training (defaults to 50)')
    corpora_group.add_argument('--cache', type=int, default=100000, help='the cache size (in sentences) for corpus reading (defaults to 1000000)')
    corpora_group.add_argument('--cache_parallel', type=int, default=None, help='the cache size (in sentences) for parallel corpus reading')

    # Embeddings/vocabulary
    embedding_group = parser.add_argument_group('embeddings', 'Embedding related arguments; either give pre-trained embeddings, or a vocabulary and embedding dimensionality to randomly initialize them')
    embedding_group.add_argument('--word_embeddings', help='table / sentence content embedding')
    embedding_group.add_argument('--field_embeddings', help='field labels embedding')
    embedding_group.add_argument('--word_vocabulary', help='table / sentences content vocabulary')
    embedding_group.add_argument('--field_vocabulary', help='field labels vocabulary')
    embedding_group.add_argument('--word_embedding_size', type=int, default=0, help='the word embedding size')
    embedding_group.add_argument('--field_embedding_size', type=int, default=0, help='the word embedding size')
    embedding_group.add_argument('--cutoff', type=int, default=0, help='cutoff vocabulary to the given size')

    # ???
    embedding_group.add_argument('--learn_encoder_embeddings', action='store_true', help='learn the encoder embeddings instead of using the pre-trained ones')
    embedding_group.add_argument('--fixed_decoder_embeddings', action='store_true', help='use fixed embeddings in the decoder instead of learning them from scratch')
    embedding_group.add_argument('--fixed_generator', action='store_true', help='use fixed embeddings in the output softmax instead of learning it from scratch')

    # Architecture
    architecture_group = parser.add_argument_group('architecture', 'Architecture related arguments')
    architecture_group.add_argument('--layers', type=int, default=2, help='the number of encoder/decoder layers (defaults to 2)')
    architecture_group.add_argument('--hidden', type=int, default=600, help='the number of dimensions for the hidden layer (defaults to 600)')
    architecture_group.add_argument('--disable_bidirectional', action='store_true', help='use a single direction encoder')
    architecture_group.add_argument('--disable_denoising', action='store_true', help='disable random swaps')
    architecture_group.add_argument('--disable_backtranslation', action='store_true', help='disable backtranslation')

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
    logging_group.add_argument('--validation', nargs='+', default=(), help='use parallel corpora for validation')
    logging_group.add_argument('--validation_directions', nargs='+', default=['src2src', 'trg2trg', 'src2trg', 'trg2src'], help='validation directions')
    logging_group.add_argument('--validation_output', metavar='PREFIX', help='output validation translations with the given prefix')
    logging_group.add_argument('--validation_beam_size', type=int, default=0, help='use beam search for validation')
    logging_group.add_argument('--print_level', type=str, default='info', help='logging level [debug | info]')

    # Other
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')

    # Parse arguments
    args = parser.parse_args()

    logger = logging.getLogger()
    if args.print_level == 'debug':
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    print("Log every %d intervals" % args.log_interval)
    # args = parser.parse_args(['--src_corpus_params', 'table, ./data/processed_data/train/train.box',
    #                           '--trg_corpus_params', 'text, ./data/processed_data/train/train.article',
    #                           '--log_interval', '10'])  # ,
    #                           # '--word_embeddings', '',
    #                           # '--field_vocabulary', '',
    #                           # '--fixed_decoder_embeddings',
    #                           # '--fixed_generator',
    #                           # '--batch', '2',
    #                           # '--cache', '100'])


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

    # if args.word_embeddings is None and args.word_vocabulary is None or\
    #         args.field_embeddings is None and args.field_vocabulary is None:
    #     print('Either an embedding or a vocabulary file must be provided')
    #     sys.exit(-1)
    # if args.word_embeddings is None and (not args.learn_encoder_embeddings or args.fixed_decoder_embeddings or args.fixed_generator):
    #     print('Either provide pre-trained word embeddings or set to learn the encoder/decoder embeddings and generator')
    #     sys.exit(-1)
    # if args.word_embeddings is None and args.word_embedding_size == 0 or \
    #         args.field_embeddings is None and args.field_embedding_size == 0:
    #     print('Either provide pre-trained embeddings or the embedding size')
    #     sys.exit(-1)
    # if len(args.validation) % 2 != 0:
    #     print('--validation should have an even number of arguments (one pair for each validation set)')
    #     sys.exit(-1)

    # Select device
    #device = devices.gpu if args.cuda else devices.cpu
    #if not args.disable_cuda and torch.cuda.is_available():
    if torch.cuda.is_available():
        device = torch.device('cuda')
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

    # Load embedding and/or vocab
    word_dict = BpeWordDict.read_vocab(bpemb_en.words)
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

    field_dict: LabelDict = torch.load('./data/processed_data/train/field.dict')
    field_embeddings = nn.Embedding(len(field_dict), bpemb_en.dim // 2, padding_idx=field_dict.pad_index)
    nn.init.normal_(field_embeddings.weight, 0, 0.1)
    nn.init.constant_(field_embeddings.weight[field_dict.pad_index], 0)
    field_embedding_size = field_embeddings.weight.data.size()[1]
    field_embeddings = field_embeddings.to(device)
    field_embeddings.weight.requires_grad = True
    logger.debug('f_embeddings is running on cuda: %d', next(word_embeddings.parameters()).is_cuda)

    # words = field_labels = word_embeddings = field_embeddings = None
    # word_embedding_size = args.word_embedding_size
    # field_embedding_size = args.field_embedding_size
    # if args.word_vocabulary is not None:
    #     f = open(args.word_vocabulary, encoding=args.encoding, errors='surrogateescape')
    #     words = [line.strip() for line in f.readlines()]
    #     if args.cutoff > 0:
    #         words = words[:args.cutoff]
    #     word_dict = data.Dictionary(words)
    # if args.field_vocabulary is not None:
    #     f = open(args.field_vocabulary, encoding=args.encoding, errors='surrogateescape')
    #     field_labels = [line.strip() for line in f.readlines()]
    #     if args.cutoff > 0:
    #         field_labels = field_labels[:args.cutoff]
    #     field_dict = data.Dictionary(field_labels)
    # if args.word_embeddings is not None:
    #     f = open(args.word_embeddings, encoding=args.encoding, errors='surrogateescape')
    #     word_embeddings, word_dict = data.read_embeddings(f, args.cutoff, words)
    #     word_embeddings = device(word_embeddings)
    #     word_embeddings.requires_grad = False
    #     if word_embedding_size == 0:
    #         word_embedding_size = word_embeddings.weight.data.size()[1]
    #     if word_embedding_size != word_embeddings.weight.data.size()[1]:
    #         print('Word Embedding sizes do not match')
    #         sys.exit(-1)
    # if args.field_embeddings is not None:
    #     f = open(args.field_embeddings, encoding=args.encoding, errors='surrogateescape')
    #     field_embeddings, field_dict = data.read_embeddings(f, args.cutoff, field_labels)
    #     field_embeddings = device(field_embeddings)
    #     field_embeddings.requires_grad = False
    #     if field_embedding_size == 0:
    #         field_embedding_size = field_embeddings.weight.data.size()[1]
    #     if field_embedding_size != field_embeddings.weight.data.size()[1]:
    #         print('Field Embedding sizes do not match')
    #         sys.exit(-1)

    src_encoder_word_embeddings = word_embeddings
    trg_encoder_word_embeddings = word_embeddings
    src_encoder_field_embeddings = field_embeddings
    trg_encoder_field_embeddings = field_embeddings

    src_decoder_word_embeddings = word_embeddings
    trg_decoder_word_embeddings = word_embeddings
    src_decoder_field_embeddings = field_embeddings
    trg_decoder_field_embeddings = field_embeddings

    # if args.fixed_generator:
    #     src_embedding_generator = device(EmbeddingGenerator(hidden_size=args.hidden, word_embedding_size=word_embedding_size))
    #     trg_embedding_generator = device(EmbeddingGenerator(hidden_size=args.hidden, word_embedding_size=word_embedding_size))
    #     add_optimizer(src_embedding_generator, (src2src_optimizers, trg2src_optimizers))
    #     add_optimizer(trg_embedding_generator, (trg2trg_optimizers, src2trg_optimizers))
    #     src_generator = device(WrappedEmbeddingGenerator(src_embedding_generator, src_embeddings))
    #     trg_generator = device(WrappedEmbeddingGenerator(trg_embedding_generator, trg_embeddings))
    # else:
    src_generator = LinearGenerator(args.hidden, len(word_dict), len(field_dict)).to(device)
    trg_generator = LinearGenerator(args.hidden, len(word_dict), len(field_dict)).to(device)
    logger.debug('src generator is running on cuda: %d', next(src_generator.parameters()).is_cuda)
    logger.debug('trg generator is running on cuda: %d', next(src_generator.parameters()).is_cuda)
    add_optimizer(src_generator, (src2src_optimizers, trg2src_optimizers))
    add_optimizer(trg_generator, (trg2trg_optimizers, src2trg_optimizers))

    # Build encoder
    encoder = RNNEncoder(word_embedding_size=word_embedding_size, field_embedding_size=field_embedding_size,
                                hidden_size=args.hidden, bidirectional=not args.disable_bidirectional,
                                layers=args.layers, dropout=args.dropout).to(device)
    logger.debug('encoder model is running on cuda: %d', next(encoder.parameters()).is_cuda)
    add_optimizer(encoder, (src2src_optimizers, trg2trg_optimizers, src2trg_optimizers, trg2src_optimizers))

    # Build decoders
    decoder = RNNAttentionDecoder(word_embedding_size=word_embedding_size,
                                         field_embedding_size=field_embedding_size, hidden_size=args.hidden,
                                         layers=args.layers, dropout=args.dropout, input_feeding=False).to(device)
    logger.debug('decoder model is running on cuda: %d', next(decoder.parameters()).is_cuda)
    logger.debug('attention model is running on cuda: %d', next(decoder.attention.parameters()).is_cuda)
    add_optimizer(decoder, (src2src_optimizers, trg2trg_optimizers, src2trg_optimizers, trg2src_optimizers))

    # src_decoder = device(RNNAttentionDecoder(word_embedding_size=word_embedding_size, hidden_size=args.hidden, layers=args.layers, dropout=args.dropout))
    # trg_decoder = device(RNNAttentionDecoder(word_embedding_size=word_embedding_size, hidden_size=args.hidden, layers=args.layers, dropout=args.dropout))
    # add_optimizer(src_decoder, (src2src_optimizers, trg2src_optimizers))
    # add_optimizer(trg_decoder, (trg2trg_optimizers, src2trg_optimizers))

    # Build translators
    src2src_translator = Translator(encoder_word_embeddings=src_encoder_word_embeddings,
                                    decoder_word_embeddings=src_decoder_word_embeddings,
                                    encoder_field_embeddings=src_encoder_field_embeddings,
                                    decoder_field_embeddings=src_decoder_field_embeddings,
                                    generator=src_generator,
                                    src_word_dict=word_dict, trg_word_dict=word_dict,
                                    src_field_dict=field_dict, trg_field_dict=field_dict,
                                    src_type=src_type, trg_type=src_type,
                                    encoder=encoder, decoder=decoder, w_sos_id=w_sos_id[src_type],
                                    denoising=not args.disable_denoising, device=device)
    src2trg_translator = Translator(encoder_word_embeddings=src_encoder_word_embeddings,
                                    decoder_word_embeddings=trg_decoder_word_embeddings,
                                    encoder_field_embeddings=src_encoder_field_embeddings,
                                    decoder_field_embeddings=trg_decoder_field_embeddings,
                                    generator=trg_generator,
                                    src_word_dict=word_dict, trg_word_dict=word_dict,
                                    src_field_dict=field_dict, trg_field_dict=field_dict,
                                    src_type=src_type, trg_type=trg_type,
                                    encoder=encoder, decoder=decoder, w_sos_id=w_sos_id[trg_type],
                                    denoising=not args.disable_denoising, device=device)
    trg2trg_translator = Translator(encoder_word_embeddings=trg_encoder_word_embeddings,
                                    decoder_word_embeddings=trg_decoder_word_embeddings,
                                    encoder_field_embeddings=trg_encoder_field_embeddings,
                                    decoder_field_embeddings=trg_decoder_field_embeddings,
                                    generator=trg_generator,
                                    src_word_dict=word_dict, trg_word_dict=word_dict,
                                    src_field_dict=field_dict, trg_field_dict=field_dict,
                                    src_type=trg_type, trg_type=trg_type,
                                    encoder=encoder, decoder=decoder, w_sos_id=w_sos_id[trg_type],
                                    denoising=not args.disable_denoising, device=device)
    trg2src_translator = Translator(encoder_word_embeddings=trg_encoder_word_embeddings,
                                    decoder_word_embeddings=src_decoder_word_embeddings,
                                    encoder_field_embeddings=trg_encoder_field_embeddings,
                                    decoder_field_embeddings=src_decoder_field_embeddings,
                                    generator=src_generator,
                                    src_word_dict=word_dict, trg_word_dict=word_dict,
                                    src_field_dict=field_dict, trg_field_dict=field_dict,
                                    src_type=trg_type, trg_type=src_type,
                                    encoder=encoder, decoder=decoder, w_sos_id=w_sos_id[src_type],
                                    denoising=not args.disable_denoising, device=device)

    # Build trainers
    trainers = []
    src2src_trainer = trg2trg_trainer = src2trg_trainer = trg2src_trainer = None
    srcback2trg_trainer = trgback2src_trainer = None
    if args.src_corpus_params is not None:
        f_content = open(src_corpus + '.content', encoding=args.encoding, errors='surrogateescape')
        f_labels = open(src_corpus + '.labels', encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(f_content, f_labels, max_sentence_length=args.max_sentence_length,
                                   cache_size=args.cache)
        src2src_trainer = Trainer(translator=src2src_translator, optimizers=src2src_optimizers, corpus=corpus,
                                  batch_size=args.batch)
        trainers.append(src2src_trainer)
        if not args.disable_backtranslation:
            trgback2src_trainer = Trainer(translator=trg2src_translator, optimizers=trg2src_optimizers,
                                          corpus=data.BacktranslatorCorpusReader(corpus=corpus,
                                                                                 translator=src2trg_translator),
                                          batch_size=args.batch)
            trainers.append(trgback2src_trainer)
    if args.trg_corpus_params is not None:
        f_content = open(trg_corpus + '.content', encoding=args.encoding, errors='surrogateescape')
        f_labels = open(trg_corpus + '.labels', encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(f_content, f_labels, max_sentence_length=args.max_sentence_length,
                                   cache_size=args.cache)
        trg2trg_trainer = Trainer(translator=trg2trg_translator, optimizers=trg2trg_optimizers, corpus=corpus,
                                  batch_size=args.batch)
        trainers.append(trg2trg_trainer)
        if not args.disable_backtranslation:
            srcback2trg_trainer = Trainer(translator=src2trg_translator, optimizers=src2trg_optimizers,
                                          corpus=data.BacktranslatorCorpusReader(corpus=corpus,
                                                                                 translator=trg2src_translator),
                                          batch_size=args.batch)
            trainers.append(srcback2trg_trainer)
    # if args.src2trg is not None:
    #     f1 = open(args.src2trg[0], encoding=args.encoding, errors='surrogateescape')
    #     f2 = open(args.src2trg[1], encoding=args.encoding, errors='surrogateescape')
    #     corpus = data.CorpusReader(f1, f2, max_sentence_length=args.max_sentence_length,
    #                                cache_size=args.cache if args.cache_parallel is None else args.cache_parallel)
    #     src2trg_trainer = Trainer(translator=src2trg_translator, optimizers=src2trg_optimizers, corpus=corpus,
    #                               batch_size=args.batch)
    #     trainers.append(src2trg_trainer)
    # if args.trg2src is not None:
    #     f1 = open(args.trg2src[0], encoding=args.encoding, errors='surrogateescape')
    #     f2 = open(args.trg2src[1], encoding=args.encoding, errors='surrogateescape')
    #     corpus = data.CorpusReader(f1, f2, max_sentence_length=args.max_sentence_length,
    #                                cache_size=args.cache if args.cache_parallel is None else args.cache_parallel)
    #     trg2src_trainer = Trainer(translator=trg2src_translator, optimizers=trg2src_optimizers, corpus=corpus,
    #                               batch_size=args.batch)
    #     trainers.append(trg2src_trainer)

    # Build validators
    # src2src_validators = []
    # trg2trg_validators = []
    # src2trg_validators = []
    # trg2src_validators = []
    # for i in range(0, len(args.validation), 2):
    #     src_validation = open(args.validation[i],   encoding=args.encoding, errors='surrogateescape').readlines()
    #     trg_validation = open(args.validation[i+1], encoding=args.encoding, errors='surrogateescape').readlines()
    #     if len(src_validation) != len(trg_validation):
    #         print('Validation sizes do not match')
    #         sys.exit(-1)
    #     map(lambda x: x.strip(), src_validation)
    #     map(lambda x: x.strip(), trg_validation)
    #     if 'src2src' in args.validation_directions:
    #         src2src_validators.append(Validator(src2src_translator, src_validation, src_validation, args.batch, args.validation_beam_size))
    #     if 'trg2trg' in args.validation_directions:
    #         trg2trg_validators.append(Validator(trg2trg_translator, trg_validation, trg_validation, args.batch, args.validation_beam_size))
    #     if 'src2trg' in args.validation_directions:
    #         src2trg_validators.append(Validator(src2trg_translator, src_validation, trg_validation, args.batch, args.validation_beam_size))
    #     if 'trg2src' in args.validation_directions:
    #         trg2src_validators.append(Validator(trg2src_translator, trg_validation, src_validation, args.batch, args.validation_beam_size))

    # Build loggers
    loggers = []

    loggers.append(Logger('Source to target (backtranslation)', srcback2trg_trainer, [], None, args.encoding))
    loggers.append(Logger('Target to source (backtranslation)', trgback2src_trainer, [], None, args.encoding))
    loggers.append(Logger('Source to source', src2src_trainer, [], None, args.encoding))
    loggers.append(Logger('Target to target', trg2trg_trainer, [], None, args.encoding))
    loggers.append(Logger('Source to target', src2trg_trainer, [], None, args.encoding))
    loggers.append(Logger('Target to source', trg2src_trainer, [], None, args.encoding))

    # loggers = []
    # src2src_output = trg2trg_output = src2trg_output = trg2src_output = None
    # if args.validation_output is not None:
    #     src2src_output = '{0}.src2src'.format(args.validation_output)
    #     trg2trg_output = '{0}.trg2trg'.format(args.validation_output)
    #     src2trg_output = '{0}.src2trg'.format(args.validation_output)
    #     trg2src_output = '{0}.trg2src'.format(args.validation_output)
    # loggers.append(Logger('Source to target (backtranslation)', srcback2trg_trainer, [], None, args.encoding))
    # loggers.append(Logger('Target to source (backtranslation)', trgback2src_trainer, [], None, args.encoding))
    # loggers.append(Logger('Source to source', src2src_trainer, src2src_validators, src2src_output, args.encoding))
    # loggers.append(Logger('Target to target', trg2trg_trainer, trg2trg_validators, trg2trg_output, args.encoding))
    # loggers.append(Logger('Source to target', src2trg_trainer, src2trg_validators, src2trg_output, args.encoding))
    # loggers.append(Logger('Target to source', trg2src_trainer, trg2src_validators, trg2src_output, args.encoding))

    # Method to save models
    def save_models(name):
        torch.save(src2src_translator, '{0}.{1}.src2src.pth'.format(args.save, name))
        torch.save(trg2trg_translator, '{0}.{1}.trg2trg.pth'.format(args.save, name))
        torch.save(src2trg_translator, '{0}.{1}.src2trg.pth'.format(args.save, name))
        torch.save(trg2src_translator, '{0}.{1}.trg2src.pth'.format(args.save, name))

    # Training
    for curr_iter in range(1, args.iterations + 1):
        for trainer in trainers:
            trainer.step(curr_iter, args.log_interval)

        if args.save is not None and args.save_interval > 0 and curr_iter % args.save_interval == 0:
            save_models('it{0}'.format(curr_iter))

        if curr_iter % args.log_interval == 0:
            print()
            print('STEP {0} x {1}'.format(curr_iter, args.batch))
            for logger in loggers:
                logger.log(curr_iter)

        # step += 1

    save_models('final')


class Trainer:
    def __init__(self, corpus, optimizers, translator, batch_size=50):
        self.corpus = corpus
        self.translator = translator
        self.optimizers = optimizers
        self.batch_size = batch_size
        self.reset_stats()

    def step(self, curr_iter, print_every=1):
        # Reset gradients
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # Read input sentences
        t = time.time()
        src_word, trg_word, src_field, trg_field = self.corpus.next_batch(self.batch_size)
        self.src_word_count += sum([len(sentence) + 1 for sentence in src_word])  # TODO Depends on special symbols EOS/SOS
        self.trg_word_count += sum([len(sentence) + 1 for sentence in trg_word])  # TODO Depends on special symbols EOS/SOS
        self.io_time += time.time() - t

        # Compute loss
        t = time.time()
        word_loss, field_loss = self.translator.score(src_word, trg_word, src_field, trg_field, curr_iter,
                                                      print_every=print_every, train=True)
        total_loss = word_loss + field_loss
        self.word_loss += word_loss.item()
        self.field_loss += field_loss.item()
        self.forward_time += time.time() - t

        # Backpropagate error + optimize
        t = time.time()
        total_loss.div(self.batch_size).backward()
        for optimizer in self.optimizers:
            optimizer.step()
        self.backward_time += time.time() - t

    def reset_stats(self):
        self.src_word_count = 0
        self.trg_word_count = 0
        self.io_time = 0
        self.forward_time = 0
        self.backward_time = 0
        self.word_loss = 0
        self.field_loss = 0

    def perplexity_per_word(self):
        return np.exp(self.word_loss/self.trg_word_count)

    def total_time(self):
        return self.io_time + self.forward_time + self.backward_time

    def words_per_second(self):
        return self.src_word_count / self.total_time(),  self.trg_word_count / self.total_time()


class Validator:
    def __init__(self, translator, source, reference, batch_size=50, beam_size=0):
        self.translator = translator
        self.source = source
        self.reference = reference
        self.sentence_count = len(source)
        self.reference_word_count = sum([len(data.tokenize(sentence)) + 1 for sentence in self.reference])  # TODO Depends on special symbols EOS/SOS
        self.batch_size = batch_size
        self.beam_size = beam_size

        # Sorting
        lengths = [len(data.tokenize(sentence)) for sentence in self.source]
        self.true2sorted = sorted(range(self.sentence_count), key=lambda x: -lengths[x])
        self.sorted2true = sorted(range(self.sentence_count), key=lambda x: self.true2sorted[x])
        self.sorted_source = [self.source[i] for i in self.true2sorted]
        self.sorted_reference = [self.reference[i] for i in self.true2sorted]

    def perplexity(self):
        loss = 0
        for i in range(0, self.sentence_count, self.batch_size):
            j = min(i + self.batch_size, self.sentence_count)
            word_loss, field_loss = self.translator.score(self.sorted_source[i:j], self.sorted_reference[i:j], train=False).data[0]
            loss += word_loss
        return np.exp(loss/self.reference_word_count)

    def translate(self):
        translations = []
        for i in range(0, self.sentence_count, self.batch_size):
            j = min(i + self.batch_size, self.sentence_count)
            batch = self.sorted_source[i:j]
            if self.beam_size <= 0:
                translations += self.translator.greedy(batch, train=False)
            else:
                translations += self.translator.beam_search(batch, train=False, beam_size=self.beam_size)
        return [translations[i] for i in self.sorted2true]


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
            w_loss = self.trainer.word_loss/self.trainer.batch_size / self.trainer.trg_word_count
            f_loss = self.trainer.field_loss/self.trainer.batch_size / self.trainer.trg_word_count

            print('  - Training:   {0:10.2f}   ({1:.2f}s: {2:.2f}tok/s src, {3:.2f}tok/s trg; epoch {4})'
                  .format(self.trainer.perplexity_per_word(), self.trainer.total_time(),
                          self.trainer.words_per_second()[0], self.trainer.words_per_second()[1], self.trainer.corpus.epoch))
            print('w_loss {0} f_loss {1}; io_time {2:.2f}s; fw_time {3:.2f}s; bw_time {4:.2f}s'
                  .format(w_loss, f_loss, self.trainer.io_time, self.trainer.forward_time, self.trainer.backward_time))
            self.trainer.reset_stats()
        for id, validator in enumerate(self.validators):
            t = time.time()
            perplexity = validator.perplexity()
            print('  - Validation: {0:10.2f}   ({1:.2f}s)'.format(perplexity, time.time() - t))
            if self.output_prefix is not None:
                f = open('{0}.{1}.{2}.txt'.format(self.output_prefix, id, step), mode='w',
                         encoding=self.encoding, errors='surrogateescape')
                for line in validator.translate():
                    print(line, file=f)
                f.close()
        sys.stdout.flush()
