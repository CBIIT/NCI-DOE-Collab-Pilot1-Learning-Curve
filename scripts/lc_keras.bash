#!/bin/bash

# Example:
# lc_keras.bash ctrp 0
# lc_keras.bash 0

DEVICE=$1
# export CUDA_VISIBLE_DEVICES=$3
export CUDA_VISIBLE_DEVICES=$DEVICE

src=ctrp

model="nn_reg0"
outdir=lc.out.${src}.${model}
mkdir -p $outdir
echo "Outdir $outdir"

echo "Source: $src"
echo "Model:  $model"
echo "CUDA:   $CUDA_VISIBLE_DEVICES"

data_version=July2020
dpath=data/ml.dfs/$data_version/data.$src.dd.ge/data.$src.dd.ge.parquet 
# spath=data/ml.dfs/$data_version/data.$src.dd.ge/data.$src.dd.ge.splits 

echo "dpath: $dpath"
# echo "spath: $spath"

n_splits=1
lc_sizes=10
min_size=2000
epoch=400

gout=$outdir/lc.${src}.${model}.log_scale
lc_step_scale=log

# gout=$outdir/lc.${src}.${model}.linear_scale
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
