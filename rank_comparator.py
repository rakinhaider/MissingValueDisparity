import logging
import os
import sys
import warnings
# Suppresing tensorflow warning
import pandas as pd

warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import *

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument('--distype', '-dt', default='ds_ccd',
                        choices=['ds_ccd', 'ccd', 'corr'],
                        help='Type of disparity')
    parser.add_argument('--priv-ic-prob', '-pic', default=0.1, type=float)
    parser.add_argument('--unpriv-ic-prob', '-upic', default=0.4, type=float)
    parser.add_argument('--group-shift', '-gs', default=0, type=int)
    parser.add_argument('--keep-im-prot', '-kip', default=False,
                        action='store_true',
                        help='Keep protected attribute in imputation')
    parser.add_argument('--keep-y', '-ky', default=False, action='store_true')
    parser.add_argument('--method', default='simple_imputer.mean',
                        choices=['baseline', 'drop', 'simple_imputer.mean',
                                 'iterative_imputer.mice',
                                 'iterative_imputer.missForest', 'knn_imputer',
                                 'group_imputer'])
    parser.add_argument('--test-method', '-tm', default='none',
                        choices=['none', 'train'])
    parser.add_argument('--header-only', default=False, action='store_true')
    args = parser.parse_args()

    protected = ["sex"]
    privileged_classes = [['Male']]

    LOG_FORMAT = '%(asctime)s - %(module)s - %(lineno)d - %(levelname)s \n %(message)s'
    logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)

    # Class shift is 10
    class_shift = args.delta
    if args.distype == 'corr':
        group_shift = args.group_shift
        dist = {
            'mus': {'x1': {
                0: [0, 0 + group_shift],
                1: [0 + class_shift, 0 + group_shift + class_shift]},
                'z': [0, 2]},
            'sigmas': {'x1': {0: [5, 5], 1: [5, 5]}, 'z': [1, 1]},
        }
        """
        dist = {
            'mus': {'x1': {
                0: [0, 0],
                1: [0, 0]},
                'z': [0, args.group_shift]},
            'sigmas': {'x1': {0: [5, 5], 1: [5, 5]}, 'z': [1, 1]},
        }
        """
    else:
        group_shift = args.group_shift
        dist = {'mus': {1: np.array([0 + class_shift, 0 + class_shift + group_shift]),
                        0: np.array([0, 0 + group_shift])},
                'sigmas': [5, 5]}
    alpha = args.alpha
    method = args.method
    if method == "group_imputer":
        keep_prot = True
    else:
        keep_prot = args.keep_im_prot

    estimator = get_estimator(args.estimator, args.reduce)
    keep_prot = args.reduce or (args.estimator == 'pr')
    n_samples = args.n_samples
    n_feature = args.n_feature
    test_method = None if args.test_method == 'none' else args.test_method

    variable = ('alpha', 'method')
    if args.print_header or args.header_only:
        print(get_table_row(is_header=True, variable=variable))
        if args.header_only:
            exit()

    models = {}
    compared_method = 'knn_imputer'
    for method in ['baseline', compared_method]:
        kwargs = {
            'protected_attribute_names': ['sex'], 'privileged_group': 'Male',
            'favorable_label': 1, 'classes': [0, 1],
            'sensitive_groups': ['Female', 'Male'],
            'group_shift': group_shift,
            'beta': 1, 'dist': dist, 'keep_im_prot': keep_prot,
            'alpha': alpha, 'method': method, 'verbose': False,
            'priv_ic_prob': args.priv_ic_prob,
            'unpriv_ic_prob': args.unpriv_ic_prob
        }
        logging.info(kwargs)
        train_fd, test_fd = get_synthetic_train_test_split(
            train_random_state=47, test_random_state=41, type=args.distype,
            n_samples=10000, n_features=n_feature,
            test_method=test_method, **kwargs)

        mod, _ = get_groupwise_performance(
            estimator, train_fd, test_fd, privileged=None)

        models[method] = mod

    probas = []
    test_x, test_y = get_xy(test_fd, keep_protected=True)
    model_features = test_x.columns[:-1]
    test_x['label'] = test_y
    for method in models.keys():
        mod = models[method]
        logging.info(mod.theta_)
        logging.info(mod.var_)
        pred_proba = mod.predict_proba(test_x[model_features])
        test_x[method+"_proba"] = pred_proba[:, 1]
        test_x[method+"_rank"] = test_x[method+"_proba"].rank()

    test_x.columns = [0, 1, 'sex', 'label', 'base_proba', 'base_rank', 'mean_proba', 'mean_rank']
    test_x.to_csv('rank.tsv', sep='\t')
    grouped = test_x.groupby(by=['sex', 'label'])
    stats = {}
    for tup, grp in grouped:
        # print(tup)
        # print(grp.describe())
        proba_comp = grp['mean_proba'] - grp['base_proba']
        rank_comp = grp['mean_rank'] - grp['base_rank']
        stat = [(proba_comp < 0).sum() * 100,
                (proba_comp > 0).sum() * 100, proba_comp.sum(),
                (rank_comp < 0).sum() * 100,
                (rank_comp > 0).sum() * 100, rank_comp.sum()]
        stat = [s / len(grp) for s in stat]
        stats[tup] = stat
        stat_str = ['u' if tup[0] == 0 else 'p', '-' if tup[1] == 0 else '+']
        stat_str += ["{:.2f}".format(stat[i]) for i in [0, 1]]
        stat_str += ["{:.2E}".format(stat[2])]
        stat_str += ["{:.2f}".format(stat[i]) for i in [3, 4]]
        stat_str += ["{:.2f}".format(stat[5])]
        print('\t & \t'.join(stat_str) + '\\\\')

    pd.set_option('display.max_columns', None)

    changes = pd.DataFrame(stats.values(), index=stats.keys(),
        columns=['proba_less', 'proba_great', 'proba_change',
                 'rank_less', 'rank_great', 'rank_change'])

    changes.to_csv('pred_changes_{:d}_{:s}.tsv'.format(
        group_shift, METHOD_SHORTS.get(compared_method, compared_method)), sep='\t')
