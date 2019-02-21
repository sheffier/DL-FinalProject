set -e

SRC=train.box
TRG=train.article
N_LINES=10

head -n $N_LINES $SRC.content > small.$SRC.content
head -n $N_LINES $SRC.labels > small.$SRC.labels
head -n $N_LINES $TRG.content > small.$TRG.content
head -n $N_LINES $TRG.labels > small.$TRG.labels
