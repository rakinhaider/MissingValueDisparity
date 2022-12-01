import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import argparse
from itertools import product
from utils import get_synthetic_train_test_split


def get_correlation_table(fd):
    results = pd.DataFrame([
        [s, index] for s, index in product(['*', 0, 1], [0, 1])],
        columns=['sex', 'index'])

    column = []
    df = fd.complete_df
    for i, row in results.iterrows():
        s, index = row['sex'], row['index']
        if s == '*':
            corr = df.corr()['label'][index]
        else:
            corr = df[df['sex'] == s].corr()['label'][index]
        column.append(corr)
    results['before'] = column

    column = []
    df = fd.imputed_df
    for i, row in results.iterrows():
        s, index = row['sex'], row['index']
        if s == '*':
            corr = df.corr()['label'][index]
        else:
            corr = df[df['sex'] == s].corr()['label'][index]
        column.append(corr)
    results['after'] = column
    results['change'] = results['before'] - results['after']

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datatype', '-dt', default='ds_ccd',
                        choices=['ds_ccd', 'ccd', 'corr'],
                        help='Dataset type')
    parser.add_argument('--alpha', '-a', type=float)
    parser.add_argument('--n_samples', '-n', default=10000, type=int)
    args = parser.parse_args()

    protected = ["sex"]
    privileged_classes = [['Male']]

    if args.datatype == 'corr':
        dist = {
            'mus': {'x1': {0: [0, 0], 1: [10, 10]}, 'z': [0, 2]},
            'sigmas': {'x1': {0: [5, 5], 1: [5, 5]}, 'z': [1, 1]},
        }
    else:
        # Class shift is 10
        class_shift = args.delta
        dist = {'mus': {1: np.array([10, 10]),
                        0: np.array([10 - class_shift, 10 - class_shift])},
                'sigmas': [3, 3]}
    train_fd, test_fd = get_synthetic_train_test_split(
        type=args.datatype, n_samples=args.n_samples, dist=dist,
        train_random_state=47, test_random_state=41,
        method='simple_imputer.mean'
    )

    print(train_fd.group_configs)
    print(get_correlation_table(train_fd))