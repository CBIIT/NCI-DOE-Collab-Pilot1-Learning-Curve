#!/bin/bash

# Example:
# lc_keras.bash ctrp 0

SOURCE=$1
DEVICE=$2
# export CUDA_VISIBLE_DEVICES=$3
export CUDA_VISIBLE_DEVICES=$DEVICE

model="nn_reg0"
outdir=lc.out.${SOURCE}.${model}
mkdir -p $outdir
echo "Outdir $outdir"

echo "Source: $SOURCE"
echo "Model:  $model"
echo "CUDA:   $CUDA_VISIBLE_DEVICES"

data_version=July2020
dpath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.parquet 
# spath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.splits 

echo "dpath: $dpath"
# echo "spath: $spath"

n_splits=1
lc_sizes=10
min_size=2000
epoch=400

gout=$outdir/lc.${SOURCE}.${model}.log_scale
lc_step_scale=log

# gout=$outdir/lc.${SOURCE}.${model}.linear_scale
# lc_step_scale=linear

echo "gout:  $gout"

# Train
python src/main_lc.py \
    -dp $dpath \
    --epoch $epoch \
    --batchnorm \
    --gout $gout \
    --lc_sizes $lc_sizes \
    --lc_step_scale $lc_step_scale \
    --min_size $min_size \
    --ml $model \
    --n_splits $n_splits

# Aggregate scores and plot
python src/agg_scores.py --res_dir $gout
