import logging
logging.getLogger().setLevel(logging.ERROR)
import os
import warnings
warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import itertools
from datasets import StandardCCDDataset


from utils import *

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument('--dataset', '-d', default='pima',
                        choices=['compas', 'german', 'bank', 'adult',
                                 'pima', 'heart', 'folkincome'])
    parser.add_argument('--priv-ic-prob', '-pic', default=0.1, type=float)
    parser.add_argument('--unpriv-ic-prob', '-upic', default=0.4, type=float)
    parser.add_argument('--group-shift', '-gs', default=0, type=int)
    parser.add_argument('--keep-im-prot', '-kip', default=False,
                        action='store_true',
                        help='Keep protected attribute in imputation')
    parser.add_argument('--keep-y', '-ky', default=False, action='store_true')
    parser.add_argument('--method', default='mean',
                        choices=['baseline', 'drop', 'mean',
                                 'mice', 'missForest', 'knn', 'softimpute',
                                 'nuclearnorm'])
    parser.add_argument('--test-method', '-tm', default='none',
                        choices=['none', 'train'])
    parser.add_argument('--header-only', default=False, action='store_true')
    # There is a mismatch in numbering of strategies.
    # args.strategy == 0 is strategy 2 in manuscript
    # args.strategy == 2 is strategy 1 in manuscript
    # args.strategy == 3 is strategy 3 in manuscript
    parser.add_argument('--strategy', default=2, type=int)
    args = parser.parse_args()

    protected = ["sex"]
    privileged_classes = [['Male']]

    LOG_FORMAT = '%(asctime)s - %(module)s - %(lineno)d - %(levelname)s \n %(message)s'
    logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)

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
    compare_method = METHOD_SHORT_TO_FULL[args.method]
    data = get_standard_dataset(dataset_name)
    std_train, std_test = data.split([0.8], shuffle=True, seed=41)
    test_fd = StandardCCDDataset(
        std_test, priv_ic_prob=0, unpriv_ic_prob=0, method='baseline',
        strategy=args.strategy)
    logging.info(f'Missing in test {test_fd.R.sum()}')
    for method in ['baseline', compare_method]:
        models[method] = {}
        strategy = args.strategy
        for rs in [0] + RANDOM_SEEDS:
            train_fd = StandardCCDDataset(
                std_train, priv_ic_prob=args.priv_ic_prob,
                unpriv_ic_prob=args.unpriv_ic_prob, method=method,
                strategy=strategy, random_seed=rs)
            incomplete_df = train_fd.get_incomplete_df(
                protected_attribute_names=train_fd.protected_attribute_names,
                label_names=train_fd.label_names,
                instance_names=train_fd.instance_names)
            logging.info(f"{method} {incomplete_df.describe().loc['count']}")
            logging.info(f'missing in train {train_fd.R.sum()}')

            mod, _ = get_groupwise_performance(
                estimator, train_fd, test_fd, privileged=None)

            models[method][rs] = mod

    stats = {}
    for rs in RANDOM_SEEDS:
        probas = []
        test_x, test_y = get_xy(test_fd, keep_protected=True)
        test_x[test_fd.label_names[0]] = test_y
        for method in models.keys():
            mod = models[method][rs]
            logging.info(mod.theta_)
            logging.info(mod.var_)
            pred_proba = mod.predict_proba(test_x[mod.feature_names_in_])
            test_x[method + "_proba"] = pred_proba[:, int(data.favorable_label)]
            test_x[method + "_rank"] = test_x[method + "_proba"].rank()

        logging.info(test_x.columns[:-4])
        new_columns = list(test_x.columns[:-4])
        new_columns += ['base_proba', 'base_rank', 'mean_proba', 'mean_rank']
        test_x.columns = new_columns
        logging.info(test_x.columns)
        test_x.to_csv('rank_{}.tsv'.format(dataset_name), sep='\t')
        group_condition = test_fd.protected_attribute_names + test_fd.label_names
        grouped = test_x.groupby(by=group_condition)
        for (s, y), grp in grouped:
            proba_comp = grp['mean_proba'] - grp['base_proba']
            rank_comp = grp['mean_rank'] - grp['base_rank']
            stat = [(proba_comp < 0).sum() * 100,
                    (proba_comp > 0).sum() * 100, proba_comp.sum(),
                    (rank_comp < 0).sum() * 100,
                    (rank_comp > 0).sum() * 100, rank_comp.sum()]
            stat = pd.Series([s / len(grp) for s in stat])
            stats[(s, y, rs)] = stat

    stats = pd.DataFrame(stats).transpose()
    all_s = np.unique(train_fd.protected_attributes, return_counts=False)
    all_y = [train_fd.favorable_label, train_fd.unfavorable_label]
    for s, y in itertools.product(all_s, all_y):
        stat_str = ['&\t({}, {})'.format(
            'u' if s == 0 else 'p',
            '+' if y == train_fd.favorable_label else '-')]
        stat = stats.loc[s, y, :]
        stat_str += [f"{stat[0].mean():.2f} ({stat[0].std():.2f})"]
        stat_str += [f"{stat[1].mean():.2f} ({stat[1].std():.2f})"]
        stat_str += [f"{stat[2].mean():.1e}"]
        stat_str += [f"{stat[3].mean():.2f} ({stat[3].std():.2f})"]
        stat_str += [f"{stat[4].mean():.2f} ({stat[4].std():.2f})"]
        stat_str += [f"{stat[5].mean():.2f}"]
        print('\t & \t'.join(stat_str) + '\\\\')
    pd.set_option('display.max_columns', None)
    stats.columns=['proba_less', 'proba_great', 'proba_change',
                   'rank_less', 'rank_great', 'rank_change']

    out_dir = f'outputs/standard/{dataset_name}/pred_changes'
    os.makedirs(out_dir, exist_ok=True)

    stats.to_csv('{:s}/pred_changes_{:s}_{:d}.tsv'.format(
        out_dir, dataset_name, strategy), sep='\t')
