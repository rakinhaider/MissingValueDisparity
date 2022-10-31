import os
import warnings
# Suppresing tensorflow warning
warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import *

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument('--distype', '-dt', default='ds_ccd',
                        choices=['ds_ccd', 'ccd'], help='Type of disparity')
    parser.add_argument('--keep-im-prot', '-kip', default=False,
                        action='store_true',
                        help='Keep protected attribute in imputation')
    parser.add_argument('--method', default='simple_imputer.mean',
                        choices=['baseline', 'drop', 'simple_imputer.mean',
                                 'iterative_imputer.mice',
                                 'iterative_imputer.missForest', 'knn_imputer',
                                 'group_imputer'])
    parser.add_argument('--header-only', default=False, action='store_true')
    args = parser.parse_args()

    protected = ["sex"]
    privileged_classes = [['Male']]

    # Class shift is 10
    class_shift = args.delta
    dist = {'mus': {1: np.array([10, 10]),
                    0: np.array([10 - class_shift, 10 - class_shift])},
            'sigmas': [3, 3]}
    alpha = args.alpha
    method = args.method
    if method == "group_imputer":
        keep_prot = True
    else:
        keep_prot = args.keep_im_prot

    kwargs = {
        'protected_attribute_names': ['sex'], 'privileged_group': 'Male',
        'favorable_class': 1, 'classes': [0, 1],
        'sensitive_groups': ['Female', 'Male'], 'group_shift': 2,
        'beta': 1, 'dist': dist, 'keep_im_prot': keep_prot,
        'alpha': alpha, 'method': method, 'verbose': False,
        'priv_ic_prob': 0.1, 'unpriv_ic_prob': 0.4
    }

    estimator = get_estimator(args.estimator, args.reduce)
    keep_prot = args.reduce or (args.estimator == 'pr')
    n_samples = args.n_samples
    n_feature = args.n_feature

    variable = ('alpha', 'method')
    if args.print_header or args.header_only:
        print_table_row(is_header=True, variable=variable)
        if args.header_only:
            exit()
    # TODO: ############ Results not matching with notebooks ##############
    train_fd, test_fd = get_datasets(
        train_random_state=47, test_random_state=41, type=args.distype,
        n_samples=n_samples, n_features=n_feature, **kwargs)
    pmod, pmod_results = get_groupwise_performance(train_fd, test_fd,
                                                   estimator,
                                                   privileged=True,
                                                   pos_rate=False)
    umod, umod_results = get_groupwise_performance(train_fd, test_fd,
                                                   estimator,
                                                   privileged=False,
                                                   pos_rate=False)
    mod, mod_results = get_groupwise_performance(train_fd, test_fd,
                                                 estimator,
                                                 privileged=None,
                                                 pos_rate=False)

    p_perf = get_model_performances(pmod, test_fd,
                                    get_predictions, keep_prot=keep_prot)
    u_perf = get_model_performances(umod, test_fd,
                                    get_predictions, keep_prot=keep_prot)
    m_perf = get_model_performances(mod, test_fd,
                                    get_predictions, keep_prot=keep_prot)
    print_table_row(is_header=False, var_value=(alpha, method),
                    p_perf=p_perf, u_perf=u_perf, m_perf=m_perf,
                    variable=variable)
