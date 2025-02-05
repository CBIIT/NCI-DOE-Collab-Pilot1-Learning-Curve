import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time
from pprint import pformat
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

# Utils
from utils.classlogger import Logger
from utils.utils import load_data, dump_dict, get_print_func
from ml.scale import scale_fea
from ml.data import extract_subset_fea

import learningcurve as lc
from learningcurve.lrn_crv import LearningCurve

# File path
fdir = Path(__file__).resolve().parent


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate learning curves (LCs).')
    # Input data
    parser.add_argument('-dp', '--datapath',
                        required=True,
                        default=fdir/"../data/ml.dfs/July2020/data.ctrp.dd.ge/data.ctrp.dd.ge.parquet",
                        type=str,
                        help='Full path to data (default: None).')
    # Pre-computed data splits
    parser.add_argument('-sd', '--splitdir',
                        default=None,
                        type=str,
                        help='Full path to data splits (default: None).')
    parser.add_argument('--split_id',
                        default=0,
                        type=int,
                        help='Split id (default: 0).')
    # Number of data splits (generated on the fly)
    parser.add_argument('--n_splits',
                        default=1,
                        type=int,
                        help='Number of splits for which to generate LCs (computed automatically) .\
                        Used only if pre-computed splits not provided (default: 3).')
    # Out dirs
    parser.add_argument('--gout',
                        default=None,
                        type=str,
                        help='Gloabl outdir to dump all the results (i.e., splits/runs) (default: None).')
    parser.add_argument('--rout',
                        default=None,
                        type=str,
                        help='Run outdir specific run/split (default: None).')
    # Target to predict
    parser.add_argument('-t', '--trg_name',
                        default='AUC',
                        type=str,
                        choices=['AUC'],
                        help='Name of target variable (default: AUC).')
    # Feature types
    parser.add_argument('-fp', '--fea_prfx',
                        nargs='+',
                        default=['ge', 'dd'],
                        choices=['DD', 'GE', 'dd', 'ge'],
                        help='Prefix to identify the features (default: [ge, dd]).')
    parser.add_argument('-fs', '--fea_sep',
                        default='_',
                        choices=['.', '_'],
                        help="Separator btw fea prefix and fea name (default: '_').")
    # Feature scaling
    parser.add_argument('-sc', '--scaler',
                        default='stnd',
                        type=str,
                        choices=['stnd', 'minmax', 'rbst'],
                        help='Feature normalization method (stnd, minmax, rbst) (default: None).')
    # Learning curve
    parser.add_argument('--lc_step_scale',
                        default='log',
                        type=str,
                        choices=['log', 'linear'],
                        help='Scale of progressive sampling of subset sizes in a LC (log, linear) (default: log).')
    parser.add_argument('--min_size',
                        default=512,
                        type=int,
                        help='The lower bound for the subset size (default: 512).')
    parser.add_argument('--max_size',
                        default=None,
                        type=int,
                        help='The upper bound for the subset size (default: None).')
    parser.add_argument('--lc_sizes',
                        default=5,
                        type=int,
                        help='Number of subset sizes (default: 5).')
    parser.add_argument('--lc_sizes_arr',
                        nargs='+',
                        type=int,
                        default=None,
                        help='List of the actual sizes in the LC plot (default: None).')
    # Model
    parser.add_argument('--ml',
                        default='lgb', type=str,
                        choices=['lgb', 'nn_reg0', 'nn_reg1'],
                        help='Choose ML model (default: lgb).')
    # NN HPs
    parser.add_argument('--epoch', default=1, type=int,
                        help='Epochs (default: 1).')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size (default: 32).')
    parser.add_argument('--dr_rate', default=0.2, type=float,
                        help='Dropout rate (default: 0.2).')
    parser.add_argument('--batchnorm', action='store_true',
                        help='Use batchnorm (default: False).')
    parser.add_argument('--opt', default='adam', type=str, choices=['sgd', 'adam'],
                        help='Optimizer (default: adam).')
    parser.add_argument('--lr', default='0.001', type=float,
                        help='Learning rate (default: 0.001).')
    # LGBM HPs
    parser.add_argument('--n_estimators', default=100, type=int,
                        help='n_estimators (default: 100).')
    parser.add_argument('--num_leaves', default=32, type=int,
                        help='num_leaves (default: 32).')
    parser.add_argument('--max_depth', default=-1, type=int,
                        help='max_depth (default: -1).')
    parser.add_argument('--learning_rate', default=0.1, type=float,
                        help='learning_rate (default: 0.1).')
    # Other
    parser.add_argument('--n_jobs', default=4, type=int, help='Default: 4.')

    args = parser.parse_args(args)
    return args


def run(args):
    # import pdb; pdb.set_trace()
    t0 = time()
    datapath = Path(args['datapath']).resolve()

    if args['max_size'] is not None:
        assert args['min_size'] < args['max_size'], f"min train size (min_size={args['min_size']}) "\
                                                    f"must be smaller than max train size "\
                                                    f"(max_size={args['max_size']})."

    if args['splitdir'] is None:
        splitdir = None
    else:
        splitdir = Path(args['splitdir']).resolve()
    split_id = args['split_id']

    # -----------------------------------------------
    #       Global outdir
    # -----------------------------------------------
    if args['gout'] is not None:
        gout = Path(args['gout']).resolve()
    else:
        gout = fdir.parent/'lc.trn'
        gout = gout/datapath.with_suffix('.lc').name
    args['gout'] = str(gout)
    os.makedirs(gout, exist_ok=True)

    # -----------------------------------------------
    #       Run (single split) outdir
    # -----------------------------------------------
    if args['rout'] is not None:
        rout = gout/args['rout']
    else:
        if splitdir is None:
            rout = gout/'run_0'
        else:
            rout = gout/f'split_{split_id}'
    args['rout'] = str(rout)
    os.makedirs(rout, exist_ok=True)

    # -----------------------------------------------
    #       Logger
    # -----------------------------------------------
    lg = Logger(rout/'lc.log')
    print_fn = get_print_func(lg.logger)
    print_fn(f'File path: {fdir}')
    print_fn(f'\n{pformat(args)}')
    dump_dict(args, outpath=rout/'trn.args.txt')

    # -----------------------------------------------
    #       Load data
    # -----------------------------------------------
    print_fn('\nLoad master dataset.')
    data = load_data(datapath)
    print_fn('data.shape {}'.format(data.shape))

    # Get features (x), target (y), and meta
    fea_list = args['fea_prfx']
    fea_sep = args['fea_sep']
    xdata = extract_subset_fea(data, fea_list=fea_list, fea_sep=fea_sep)
    meta = data.drop(columns=xdata.columns)
    ydata = meta[[ args['trg_name'] ]]
    del data

    # -----------------------------------------------
    #       Scale features
    # -----------------------------------------------
    xdata = scale_fea(xdata=xdata, scaler_name=args['scaler'])

    # -----------------------------------------------
    #       Data splits
    # -----------------------------------------------
    if splitdir is None:
        cv_lists = None
    else:
        split_pattern = f'1fold_s{split_id}_*_id.csv'
        single_split_files = glob(str(splitdir/split_pattern))

        # Get indices for the split
        for id_file in single_split_files:
            if 'tr_id' in id_file:
                tr_id = load_data(id_file).values.reshape(-1,)
            elif 'vl_id' in id_file:
                vl_id = load_data(id_file).values.reshape(-1,)
            elif 'te_id' in id_file:
                te_id = load_data(id_file).values.reshape(-1,)

        cv_lists = (tr_id, vl_id, te_id)

    # -----------------------------------------------
    #      ML model configs
    # -----------------------------------------------
    if args['ml'] == 'lgb':
        # LGBM regressor model definition
        import lightgbm as lgb
        framework = 'lightgbm'
        ml_model_def = lgb.LGBMRegressor
        mltype = 'reg'

        ml_init_kwargs = {'n_estimators': args['n_estimators'],
                          'max_depth': args['max_depth'],
                          'learning_rate': args['learning_rate'],
                          'num_leaves': args['num_leaves'],
                          'n_jobs': args['n_jobs'],
                          'random_state': None}
        ml_fit_kwargs = {'verbose': False, 'early_stopping_rounds': 10}
        data_prep_def = None
        keras_callbacks_def = None
        keras_clr_kwargs = None

    elif args['ml'] == 'nn_reg0':
        # Keras model def
        from models.keras_model import (nn_reg0_model_def, data_prep_nn0_def,
                                        model_callback_def)
        framework = 'keras'
        mltype = 'reg'
        keras_callbacks_def = model_callback_def
        data_prep_def = data_prep_nn0_def

        if (args['ml'] == 'nn_reg0'):
            ml_model_def = nn_reg0_model_def

        ml_init_kwargs = {'input_dim': xdata.shape[1],
                          'dr_rate': args['dr_rate'],
                          'opt_name': args['opt'],
                          'lr': args['lr'],
                          'batchnorm': args['batchnorm']}
        ml_fit_kwargs = {'epochs': args['epoch'],
                         'batch_size': args['batch_size'],
                         'verbose': 1}
        keras_clr_kwargs = {}

    elif args['ml'] == 'nn_reg1':
        from models.keras_model import (nn_reg1_model_def, data_prep_nn1_def,
                                        model_callback_def)
        framework = 'keras'
        mltype = 'reg'
        keras_callbacks_def = model_callback_def
        data_prep_def = data_prep_nn1_def

        if (args['ml'] == 'nn_reg1'):
            ml_model_def = nn_reg1_model_def

        x_ge = extract_subset_fea(xdata, fea_list=['ge'], fea_sep='_')
        x_dd = extract_subset_fea(xdata, fea_list=['dd'], fea_sep='_')

        ml_init_kwargs = {'in_dim_ge': x_ge.shape[1],
                          'in_dim_dd': x_dd.shape[1],
                          'dr_rate': args['dr_rate'],
                          'opt_name': args['opt'],
                          'lr': args['lr'],
                          'batchnorm': args['batchnorm']}
        ml_fit_kwargs = {'epochs': args['epoch'],
                         'batch_size': args['batch_size'],
                         'verbose': 1}
        keras_clr_kwargs = {}
        del x_ge, x_dd

    # Print NN
    if len(ml_init_kwargs) and ('nn' in args['ml']):
        model = ml_model_def(**ml_init_kwargs)
        model.summary(print_fn=lg.logger.info)
        del model

    # -----------------------------------------------
    #      Learning curve
    # -----------------------------------------------
    # LC args
    lc_init_args = {'cv_lists': cv_lists,
                    'n_splits': args['n_splits'],
                    'mltype': mltype,
                    'lc_step_scale': args['lc_step_scale'],
                    'lc_sizes': args['lc_sizes'],
                    'min_size': args['min_size'],
                    'max_size': args['max_size'],
                    'lc_sizes_arr': args['lc_sizes_arr'],
                    'outdir': rout,
                    'print_fn': print_fn}

    lc_trn_args = {'framework': framework,
                   'n_jobs': args['n_jobs'],
                   'ml_model_def': ml_model_def,
                   'ml_init_args': ml_init_kwargs,
                   'ml_fit_args': ml_fit_kwargs,
                   'data_prep_def': data_prep_def,
                   'keras_callbacks_def': keras_callbacks_def,
                   'keras_clr_args': keras_clr_kwargs}

    # LC object
    lc_obj = LearningCurve(X=xdata, Y=ydata, meta=meta, **lc_init_args)
    lc_scores = lc_obj.trn_learning_curve(**lc_trn_args)

    # Dump all scores
    lc_scores.to_csv(rout/'lc_scores.csv', index=False)

    # Dump args
    dump_dict(args, outpath=rout/'args.txt')

    # ------------------------------------------------------
    if (time()-t0)//3600 > 0:
        print_fn('Runtime: {:.1f} hrs'.format((time()-t0)/3600))
    else:
        print_fn('Runtime: {:.1f} mins'.format((time()-t0)/60))

    print_fn('Done.')
    lg.close_logger()
    del xdata, ydata

    return None


def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)
    return None


if __name__ == '__main__':
    main(sys.argv[1:])
