Unsupervised Table-2-Text
==============

This repository contains contains the implementation of the final project of the [Deep Learning](https://www30.tau.ac.il/yedion/syllabus.asp?course=0368448801) course taken at Fall 2018.
This README provides code explanation and usage.


## Dependencies

* Python 3
* [NumPy](http://www.numpy.org/)
* [PyTorch](http://pytorch.org/) (currently tested on version 1.0)
* [BPEmb](https://github.com/bheinzerling/bpemb) (generate and apply BPE codes)
* [tensorboardX](https://github.com/lanpa/tensorboardX) (for logging events during training)
* [tqdm](https://github.com/tqdm/tqdm)

### Download / preprocess data

The first thing to do to run the Table-2-Text model is to download and preprocess data. To do so, just run:

```
git clone ENTER MY REPO
cd REPO_NAME
./get_data.sh
```


### Train the model

```
usage: train.py [-h] [--src_corpus_params SRC_CORPUS_PARAMS]
                [--trg_corpus_params TRG_CORPUS_PARAMS]
                [--src_para_corpus_params SRC_PARA_CORPUS_PARAMS]
                [--trg_para_corpus_params TRG_PARA_CORPUS_PARAMS]
                [--corpus_mode CORPUS_MODE]
                [--max_sentence_length MAX_SENTENCE_LENGTH] [--cache CACHE]
                [--emb_dim EMB_DIM] [--word_vocab_size WORD_VOCAB_SIZE]
                [--layers LAYERS] [--hidden HIDDEN] [--dis_hidden DIS_HIDDEN]
                [--n_dis_layers N_DIS_LAYERS] [--disable_bidirectional]
                [--disable_backtranslation] [--disable_field_loss]
                [--disable_discriminator] [--shared_enc] [--shared_dec]
                [--denoising_mode DENOISING_MODE]
                [--word_shuffle WORD_SHUFFLE] [--word_dropout WORD_DROPOUT]
                [--word_blank WORD_BLANK] [--batch BATCH]
                [--learning_rate LEARNING_RATE] [--dropout PROB]
                [--param_init RANGE] [--iterations ITERATIONS]
                [--beam_size BEAM_SIZE] [--save PREFIX]
                [--save_interval SAVE_INTERVAL] [--log_interval LOG_INTERVAL]
                [--dbg_print_interval DBG_PRINT_INTERVAL]
                [--src_valid_corpus SRC_VALID_CORPUS]
                [--trg_valid_corpus TRG_VALID_CORPUS]
                [--print_level PRINT_LEVEL] [--metadata_path METADATA_PATH]
                [--encoding ENCODING] [--cuda CUDA]
                [--bleu_device BLEU_DEVICE]
```

### Evaluating a trained model


```
usage: translate.py [-h] [--model_list MODEL_LIST] [--batch_size BATCH_SIZE]
                    [--encoding ENCODING] [--testset_path TESTSET_PATH]
                    [--device_list DEVICE_LIST] [--prefix PREFIX]
                    [--train_corpus_mode TRAIN_CORPUS_MODE]
                    [--direction DIRECTION] [--log_dir LOG_DIR]
                    [--translation_dir TRANSLATION_DIR]

```

For more details and additional options, run the above scripts with the `--help` flag.
