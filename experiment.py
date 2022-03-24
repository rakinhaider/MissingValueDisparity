import os
import numpy as np
import warnings
from utils import *

# Suppresing tensorflow warning
warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    args = get_parser().parse_args()

    protected = ["sex"]
    privileged_classes = [['Male']]

    # Class shift is 10
    class_shift = 2
    dist = {'mus': {1: np.array([10, 15]),
                    0: np.array([10 - class_shift, 15 - class_shift])},
            'sigmas': [3, 3]}

    kwargs = {
        'protected_attribute_names': ['sex'],
        'privileged_group': 'Male',
        'favorable_class': 1,
        'classes': [0, 1],
        'sensitive_groups': ['Female', 'Male'],
        'group_shift': 2,
        'alpha': 0.5, 'beta': 1,
        'dist': dist
    }

    estimator = get_estimator(args.estimator, args.reduce)
    keep_prot = args.reduce or (args.estimator == 'pr')
    n_samples = args.n_samples
    n_feature = args.n_feature

    print_table_row(is_header=True, variable='method')
    for method in ['knn_imputer', 'simple_imputer.mean', 'iterative_imputer.mice',
                   'iterative_imputer.missForest']:
        kwargs['method'] = method
        kwargs['verbose'] = False
        train_fd, test_fd = get_datasets(train_random_state=43,
                                         test_random_state=43,
                                         n_samples=n_samples,
                                         n_features=n_feature, **kwargs)
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
        print_table_row(is_header=False, var_value=method, p_perf=p_perf,
                        u_perf=u_perf, m_perf=m_perf, variable='method')
