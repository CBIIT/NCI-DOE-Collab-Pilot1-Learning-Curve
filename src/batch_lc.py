"""
A batch prcoessing that calls main_lc.py with the same set of parameters
but different split ids.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from glob import glob

import numpy as np
# from joblib import Parallel, delayed

fdir = Path(__file__).resolve().parent
import main_lc


def run_split_fly(n_splits, rout, *args):
    """ Generate split on the fly. """
    print('Calling run_split_fly ...')
    main_lc.main([ '--n_splits', str(n_splits), '--rout', 'run_'+str(rout), *args ])


parser = argparse.ArgumentParser()
parser.add_argument('-ns', '--n_splits',
                    default=10,
                    type=int,
                    help='Use a subset of splits (default: 10).')
args, other_args = parser.parse_known_args()
print(args)


main_fn = run_split_fly
splits_arr = [1 for _ in range(args.n_splits)]
runs_arr = np.arange(args.n_splits)
n_splits = args.n_splits


# Main execution
for s, r in zip( splits_arr[:n_splits], runs_arr[:n_splits] ):
    print(f'Processing split {s}')
    other_args_run = other_args.copy()
    main_fn(s, r, *other_args_run) # only one split for every run

print('Done.')