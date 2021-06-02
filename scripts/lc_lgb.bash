#!/bin/bash

# Example:
# bash scripts/lc_lgb.bash ctrp 16 none
# bash scripts/lc_lgb.bash gdsc1 16 none
# bash scripts/lc_lgb.bash gdsc2 16 none

SOURCE=$1

model="lgb"
outdir=lc.out.${SOURCE}.${model}
mkdir -p $outdir
echo "Outdir $outdir"

echo "Source: $SOURCE"
echo "Model:  $model"

data_version=July2020
dpath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.parquet
# spath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge/data.$SOURCE.dd.ge.splits

echo "dpath: $dpath"
# echo "spath: $spath"

n_splits=1
lc_sizes=10
min_size=1000
n_jobs=8

gout=$outdir/lc.${SOURCE}.${model}.log_scale
lc_step_scale=log

# gout=$outdir/lc.${SOURCE}.${model}.linear_scale
# lc_step_scale=linear

echo "gout:  $gout"

# echo "gout:  $gout"
# python src/batch_lc.py \
#     -dp $dpath \
#     --ml $model \
#     --gout $gout \
#     --lc_sizes $lc_sizes \
#     --min_size $min_size \
#     --n_jobs 8 \
#     --n_splits $n_splits \
#     --par_jobs 4 \
#     --lc_step_scale $lc_step_scale

# Train
python src/main_lc.py \
    -dp $dpath \
    --gout $gout \
    --lc_sizes $lc_sizes \
    --lc_step_scale $lc_step_scale \
    --min_size $min_size \
    --ml $model \
    --n_jobs $n_jobs \
    --n_splits $n_splits

# Aggregate scores and plot
python src/agg_scores.py --res_dir $gout
