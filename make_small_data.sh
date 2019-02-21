set -e

DATA_PATH=$PWD/data/processed_data/train
SRC=$DATA_PATH/train.box
TRG=$DATA_PATH/train.article
N_LINES=100

head -n $N_LINES $SRC.content > $SRC.small.content
head -n $N_LINES $SRC.labels > $SRC.small.labels
head -n $N_LINES $TRG.content > $TRG.small.content
head -n $N_LINES $TRG.labels > $TRG.small.labels
