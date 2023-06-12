import numpy as np
import os
import logging
import warnings
import scipy.stats as stats
from sklearn.metrics import ndcg_score

warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import utils
from utils_ranking import conditional_ndcg, ndd


def relevance_scores(df, inplace=True):
    df['relevance'] = np.zeros(len(df), dtype=float)
    for s in [0, 1]:
        grp = df[df['sex'] == s]
        percentile_prod = np.ones(len(grp), dtype=float)
        for col in ['0', '1']:
            relevance_scores = stats.percentileofscore(grp[col], grp[col]) / 100
            percentile_prod *= relevance_scores
        df.loc[grp.index, 'relevance'] = percentile_prod

    maximum, minimum = df['relevance'].max(), df['relevance'].min()
    df['relevance'] = (df['relevance'] - minimum) / (maximum - minimum)
    assert df['relevance'].max() == 1
    assert df['relevance'].min() == 0
    if not inplace:
        return df


if __name__ == "__main__":
    parser = utils.get_parser()
    parser.add_argument('--distype', '-dt', default='ccd',
                        choices=['ds_ccd', 'ccd', 'corr'],
                        help='Type of disparity')
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
    else:
        group_shift = args.group_shift
        dist = {'mus': {
            1: np.array([0 + class_shift, 0 + class_shift + group_shift]),
            0: np.array([0, 0 + group_shift])},
            'sigmas': [5, 5]}
    alpha = args.alpha
    method = args.method
    if method == "group_imputer":
        keep_prot = True
    else:
        keep_prot = args.keep_im_prot

    estimator = utils.get_estimator(args.estimator, args.reduce)
    keep_prot = args.reduce or (args.estimator == 'pr')
    n_samples = args.n_samples
    n_feature = args.n_feature
    test_method = None if args.test_method == 'none' else args.test_method

    models = {}
    method_full = utils.METHOD_SHORT_TO_FULL[args.method]
    for m in ['baseline', method_full]:
        kwargs = {
            'protected_attribute_names': ['sex'], 'privileged_group': 'Male',
            'favorable_label': 1, 'classes': [0, 1],
            'sensitive_groups': ['Female', 'Male'],
            'group_shift': group_shift,
            'beta': 1, 'dist': dist, 'keep_im_prot': keep_prot,
            'alpha': alpha, 'method': m, 'verbose': False,
            'priv_ic_prob': args.priv_ic_prob,
            'unpriv_ic_prob': args.unpriv_ic_prob
        }
        logging.info(kwargs)
        train_fd, test_fd = utils.get_synthetic_train_test_split(
            train_random_state=args.tr_rs, test_random_state=args.te_rs,
            type=args.distype, n_samples=10000, n_features=n_feature,
            test_method=test_method, **kwargs)

        mod, _ = utils.get_groupwise_performance(
            estimator, train_fd, test_fd, privileged=None)

        models[m] = mod

    probas = []
    test_x, test_y = utils.get_xy(test_fd, keep_protected=True)
    model_features = test_x.columns[:-1]
    test = test_x.copy()
    test['label'] = test_y
    relevance_scores(test)
    test['orig_rank'] = test['relevance'].rank()

    for method in models.keys():
        mod = models[method]
        logging.info(mod.theta_)
        logging.info(mod.var_)
        pred_proba = mod.predict_proba(test_x[model_features])
        logging.info(pred_proba[0:10])
        test[method + "_proba"] = pred_proba[:, 1]
        test[method + "_rank"] = test[method + "_proba"].rank()

    gain = test[['relevance', 'baseline_proba', f'{method_full}_proba', 'sex', 'label']]
    gain = gain.sort_values(by=['relevance'], ascending=False,
                            ignore_index=False)
    gain.set_index(keys=['sex', 'label'], append=True, inplace=True)
    orig_col = 'relevance'
    base_col = 'baseline_proba'
    method_col = f'{method_full}_proba'
    print('sklearn ndcg')
    print(ndcg_score([gain[orig_col].values], [gain[base_col].values], ignore_ties=True))
    print(ndcg_score([gain[orig_col].values], [gain[method_col].values], ignore_ties=True))

    print('My ndcg')
    print(conditional_ndcg([gain[orig_col]], [gain[base_col]]))
    print(conditional_ndcg([gain[orig_col]], [gain[method_col]]))

    print('Privileged ndcg')
    print(conditional_ndcg([gain[orig_col]], [gain[base_col]], privilege=1))
    print(conditional_ndcg([gain[orig_col]], [gain[method_col]], privilege=1))

    print('Unprivileged ndcg')
    print(conditional_ndcg([gain[orig_col]], [gain[base_col]], privilege=0))
    print(conditional_ndcg([gain[orig_col]], [gain[method_col]], privilege=0))

    for s in [0, 1]:
        for t in [0, 1]:
            base_cndcg = conditional_ndcg([gain[orig_col]], [gain[base_col]],
                                     privilege=s, target=t)
            meth_cndcg = conditional_ndcg([gain[orig_col]], [gain[method_col]],
                                     privilege=s, target=t)

            print(f'{s} {t} {base_cndcg:.6f} {meth_cndcg:.6f}')

    base_col = 'baseline_rank'
    method_col = f'{method + "_rank"}'
    ranks = test[['orig_rank', base_col, method_col, 'sex']]
    ranks.set_index(keys=['sex'], append=True, inplace=True)

    ndd_orig = ndd(ranks['orig_rank'], ranks['orig_rank'], reduce=True)
    # print(' & '.join([f'{i:.6f}' for i in ndd_orig]))
    print(ndd_orig)
    ndd_base = ndd(ranks[base_col], ranks['orig_rank'], reduce=True)
    print(ndd_base)
    # print(' & '.join([f'{i:.6f}' for i in ndd_base]))
    ndd_meth = ndd(ranks[method_col], ranks['orig_rank'], reduce=True)
    print(ndd_meth)
    # print(' & '.join([f'{i:.6f}' for i in ndd_meth]))
