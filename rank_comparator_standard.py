import logging
import os
import sys
import warnings
# Suppresing tensorflow warning
import pandas as pd

from datasets import StandardCCDDataset

warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import *

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument('--dataset', '-d', default='pima',
                        choices=['compas', 'pima'])
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
    parser.add_argument('--strategy', default=2, type=int)
    args = parser.parse_args()

    protected = ["sex"]
    privileged_classes = [['Male']]

    LOG_FORMAT = '%(asctime)s - %(module)s - %(lineno)d - %(levelname)s \n %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

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

    # TODO: ############ Results not matching with notebooks ##############
    models = {}
    # dataset_name = args.dataset
    dataset_name = args.dataset
    for method in ['baseline', 'simple_imputer.mean']:
        strategy = args.strategy
        data = get_standard_dataset(dataset_name)

        std_train, std_test = data.split([0.8], shuffle=True, seed=41)
        train_fd = StandardCCDDataset(std_train, priv_ic_prob=args.priv_ic_prob,
                                      unpriv_ic_prob=args.unpriv_ic_prob,
                                      method=method, strategy=strategy)
        incomplete_df = train_fd.get_incomplete_df(
            protected_attribute_names=train_fd.protected_attribute_names,
            label_names=train_fd.label_names, instance_names=train_fd.instance_names)
        logging.info(incomplete_df.describe().loc['count'])
        test_fd = StandardCCDDataset(std_test, priv_ic_prob=0, unpriv_ic_prob=0,
                                  method='baseline')

        pmod, p_perf = get_groupwise_performance(
            estimator, train_fd, test_fd, privileged=True)
        umod, u_perf = get_groupwise_performance(
            estimator, train_fd, test_fd, privileged=False)
        mod, m_perf = get_groupwise_performance(
            estimator, train_fd, test_fd, privileged=None)

        models[method] = dict(zip(['pmod', 'umod', 'mod'], [pmod, umod, mod]))

        row = get_table_row(
            is_header=False, var_value=(alpha, method), p_perf=m_perf,
            u_perf=m_perf, m_perf=m_perf, variable=variable)
        print(row)
        sys.stdout.flush()

    probas = []
    test_x, test_y = get_xy(test_fd, keep_protected=True)
    model_features = test_x.columns[:-1]
    test_x['label'] = test_y
    for method in models.keys():
        mod = models[method]['mod']
        logging.info(mod.theta_)
        logging.info(mod.var_)
        pred_proba = mod.predict_proba(test_x[model_features])
        test_x[method+"_proba"] = pred_proba[:, 1]
        test_x[method+"_rank"] = test_x[method+"_proba"].rank()

    logging.info(test_x.columns[:-4])
    test_x.columns = list(test_x.columns[:-4]) + ['base_proba', 'base_rank', 'mean_proba', 'mean_rank']
    logging.info(test_x.columns)
    test_x.to_csv('rank_{}.tsv'.format(dataset_name), sep='\t')
    print(train_fd.protected_attribute_names + ['label'])
    grouped = test_x.groupby(by=train_fd.protected_attribute_names + ['label'])
    stats = []
    for tup, grp in grouped:
        print(tup)
        print(grp.describe())
        proba_comp = grp['mean_proba'] - grp['base_proba']
        rank_comp = grp['mean_rank'] - grp['base_rank']
        stats.append([(proba_comp < 0).sum(), (proba_comp >= 0).sum(), proba_comp.sum(),
                      (rank_comp < 0).sum(), (rank_comp >= 0).sum(), rank_comp.sum()])

    pd.set_option('display.max_columns', None)
    changes = pd.DataFrame(stats, index=[(0, 0), (0, 1), (1, 0), (1, 1)],
                           columns=['proba_less', 'proba_great', 'proba_change',
                                    'rank_less', 'rank_great', 'rank_change'])

    changes.to_csv('pred_changes_{:s}_{:d}.tsv'.format(dataset_name, strategy), sep='\t')
