#!/bin/bash

# Shane Settle, settle.shane@gmail.com, 2018

# This script runs code which can be found
# in the Switcboard Kaldi recipe (kaldi-trunk/egs/swbd/s5c).

# with a slight alteration:
# "local/swbd1_data_prep.sh"
#   --> addition at line 94
#   --> subtraction at line 115

# and with the addition:
# "utils/apply_cmvn_stats.sh"
# "utils/get_query_list.py"
# "utils/get_intervals.py"

. cmd.sh
. path.sh


#####
nj=8  # number of jobs to run

stage=1

swbd="/share/data/speech/Datasets/switchboard/"  # local switchboard location

min_word_length=6  # minimum word length for written words
min_audio_duration=50  # minimum audio duration for spoken words
min_train_occurrence_count=2  # minimum number of occurrences in training set
queries="$PWD/data/len${min_word_length}-${min_audio_duration}frames-count${min_train_occurrence_count}"

convsides="../partitions"  # paritioning over conversation sides

#####

x=train
datadir="$PWD/data/$x"
mfccdir="$PWD/mfcc"

make_mfcc_logdir="$PWD/exp/$x/make_mfcc"
compute_cmvn_logdir="$PWD/exp/$x/compute_cmvn"
apply_cmvn_logdir="$PWD/exp/$x/apply_cmvn"
words="$datadir/words"
scp_file="$datadir/mfcc_cmvn_ddelta.scp"

#####

if [ $stage -le 1 ]; then

    local/swbd1_data_download.sh $swbd
    local/swbd1_data_prep.sh $swbd

    echo "making mfccs, out: "
    echo "output: $mfccdir"
    echo "logs: $make_mfcc_logdir"
    steps/make_mfcc.sh --nj $nj $datadir $make_mfcc_logdir $mfccdir
    echo "done"

    echo "computing cmvn statistics"
    echo "output: $mfccdir"
    echo "logs: $compute_cmvn_logdir"
    steps/compute_cmvn_stats.sh $datadir $compute_cmvn_logdir $mfccdir
    echo "done"

    echo "checking $datadir"
    utils/fix_data_dir.sh $datadir
    echo "done"

    echo "applying cmvn"
    echo "output: $mfccdir"
    echo "logs: $apply_cmvn_logdir"
    utils/apply_cmvn_stats.sh --nj $nj $datadir $apply_cmvn_logdir $mfccdir
    echo "done"

    rm $mfccdir/raw_mfcc_$x.*  # clean raw files, won't need these
    cat $mfccdir/mfcc_cmvn_ddelta_$x.*.scp > $datadir/mfcc_cmvn_ddelta.scp
fi

if [ $stage -le 2 ]; then

    for partition in "train" "dev" "test"; do  # create query lists for each partition

        if [ $partition != "train" ]; then
            min_occurrence_count=1
        else
            min_occurrence_count=$min_train_occurrence_count
        fi

        mkdir -p $queries/$partition

        python utils/get_query_list.py \
            --words $words \
            --convsides $convsides/$partition.txt \
            --min-word-length $min_word_length \
            --min-audio-duration $min_audio_duration \
            --min-occurrence-count $min_occurrence_count \
            --query-list-file $queries/$partition/list > $queries/$partition/list.log &
    done
    wait
fi

if [ $stage -le 3 ]; then
    for partition in "train" "dev" "test"; do  # create intervals for each partition
        
        mkdir -p $queries/$partition
        
        python utils/get_intervals.py \
            --query-list-file $queries/$partition/list \
            --scp-file $scp_file \
            --intervals-file $queries/$partition/intervals > $queries/$partition/intervals.log &
    done
    wait
fi


if [ $stage -le 4 ]; then
    for partition in "train" "dev" "test"; do  # create segments
        
        mkdir -p $queries/$partition
        
        extract-rows $queries/$partition/intervals "scp:$scp_file" "ark,scp:$queries/$partition/mfcc.ark,$queries/$partition/mfcc.scp" &
    done
    wait
fi

echo "finished."
