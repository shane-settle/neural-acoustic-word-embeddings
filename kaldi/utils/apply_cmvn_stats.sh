#!/bin/bash

# Shane Settle, settle.shane@gmail.com, 2018

nj=4

. utils/parse_options.sh || exit 1;

if [ $# != 3 ]; then
    echo "usage: ${0} data_dir exp_dir feat_dir"
    exit 1;
fi

datadir=$1
logdir=$2
mfccdir=$3

name=`basename $datadir`

mkdir -p $logdir/apply_cmvn_log

sdatadir=$datadir/split$nj;
mkdir -p $sdatadir

utils/split_data.sh $datadir $nj || exit 1;

utt2spk="ark:$sdatadir/JOB/utt2spk"
cmvn="scp:$sdatadir/JOB/cmvn.scp"
feats="scp:$sdatadir/JOB/feats.scp"
ddelta_out="ark,scp:$mfccdir/mfcc_cmvn_ddelta_$name.JOB.ark,$mfccdir/mfcc_cmvn_ddelta_$name.JOB.scp"

cmd="apply-cmvn --norm-vars=true --utt2spk=$utt2spk $cmvn $feats ark:- | add-deltas ark:- $ddelta_out"

$train_cmd JOB=1:$nj $logdir/cmvn_ddelta.JOB.log $cmd || exit 1;
