import logging
import os
import warnings
import pandas as pd
from tqdm import tqdm

# Suppresing tensorflow warning
logging.getLogger().setLevel(logging.ERROR)
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
    parser.add_argument('--method', default='mean',
                        choices=['baseline', 'drop', 'mean',
                                 'mice', 'missForest', 'knn', 'softimpute'])
    parser.add_argument('--test-method', '-tm', default='none',
                        choices=['none', 'train'])
    parser.add_argument('--header-only', default=False, action='store_true')
    args = parser.parse_args()

    protected = ["sex"]
    privileged_classes = [['Male']]

    LOG_FORMAT = '%(asctime)s - %(module)s - %(lineno)d - %(levelname)s \n %(message)s'
    logging.basicConfig(level=logging.WARN, format=LOG_FORMAT)

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
        dist = {'mus': {1: np.array([0 + class_shift,
                                     0 + class_shift + args.group_shift]),
                        0: np.array([0, 0 + args.group_shift])},
                'sigmas': [5, 5]}
    alpha = args.alpha
    method = METHOD_SHORT_TO_FULL[args.method]
    if method == "group_imputer":
        keep_prot = True
    else:
        keep_prot = args.keep_im_prot
    kwargs = {
        'protected_attribute_names': ['sex'], 'privileged_group': 'Male',
        'favorable_label': 1, 'classes': [0, 1],
        'sensitive_groups': ['Female', 'Male'], 'group_shift': args.group_shift,
        'beta': 1, 'dist': dist, 'keep_im_prot': keep_prot,
        'alpha': alpha, 'method': method, 'verbose': False,
        'priv_ic_prob': args.priv_ic_prob, 'unpriv_ic_prob': args.unpriv_ic_prob
    }
    estimator = get_estimator(args.estimator, args.reduce)
    keep_prot = args.reduce or (args.estimator == 'pr')
    n_samples = args.n_samples
    n_feature = args.n_feature
    test_method = None if args.test_method == 'none' else args.test_method

    variable = ('method')
    if args.print_header or args.header_only:
        print(get_table_row(is_header=True, variable=variable))
        if args.header_only:
            exit()

    logging.info(kwargs)

    m_perfs = []
    train_fd, test_fd = get_synthetic_train_test_split(
        train_random_state=47, test_random_state=41, type=args.distype,
        n_samples=n_samples, n_features=n_feature,
        test_method=test_method, **kwargs)

    pmod, p_perf = get_groupwise_performance(
        estimator, train_fd, test_fd, privileged=True)
    umod, u_perf = get_groupwise_performance(
        estimator, train_fd, test_fd, privileged=False)
    mod, m_perf = get_groupwise_performance(
        estimator, train_fd, test_fd, privileged=None)

    m_perfs.append(m_perf)
    row = get_table_row(is_header=False, var_value=(args.method),
                        m_perf=m_perf, variable=variable)

    m_perfs = pd.DataFrame(m_perfs)
    m_perfs = m_perfs[["AC_p", "AC_u", "SR_p", "SR_u", "FPR_p", "FPR_u"]]

    stat_strs = [f'{method}', f'{args.priv_ic_prob}', f'{args.unpriv_ic_prob}']
    for s in ["AC", "SR", "FPR"]:
        diffs = m_perfs[f'{s}_p'] - m_perfs[f'{s}_u']
        stat_strs += [f'{m_perfs.iloc[0][f"{s}_p"]:0.2f}', f'{m_perfs.iloc[0][f"{s}_u"]:0.2f}']
    print("\t".join(stat_strs))