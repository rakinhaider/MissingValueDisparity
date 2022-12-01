import argparse
import os
import random as rnd
import warnings
# Suppresing tensorflow warning
import numpy as np
import logging

warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
from datasets.standard_ccd_dataset import StandardCCDDataset

if __name__ == "__main__":
    LOG_FORMAT = '%(asctime)s - %(module)s - %(lineno)d - %(levelname)s \n %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['compas', 'bank', 'german'])
    parser.add_argument('--method', default='simple_imputer.mean',
                        choices=['baseline', 'drop', 'simple_imputer.mean',
                                 'iterative_imputer.mice',
                                 'iterative_imputer.missForest', 'knn_imputer'])
    parser.add_argument('--print-header', default=False, action='store_true')
    parser.add_argument('--header-only', default=False, action='store_true')

    parser.add_argument('--priv-ic-prob', '-pic', default=0.1, type=float)
    parser.add_argument('--unpriv-ic-prob', '-upic', default=0.4, type=float)

    parser.add_argument('--random-seed', default=41, type=int)

    args = parser.parse_args()
    args.reduce = False
    method = args.method
    estimator = get_estimator('cat_nb', False)
    # keep_prot = args.reduce or (args.estimator == 'pr')
    keep_prot = False

    variable = 'dataset'
    if args.print_header or args.header_only:
        print(get_table_row(is_header=True, variable=variable))
        if args.header_only:
            exit()

    np.random.seed(args.random_seed)

    data = get_standard_dataset(args.dataset)
    df, _ = data.convert_to_dataframe()
    train, test = data.split([0.8], shuffle=True, seed=args.random_seed)
    logging.info('train')
    train = StandardCCDDataset(train, priv_ic_prob=args.priv_ic_prob,
                               unpriv_ic_prob=args.unpriv_ic_prob,
                               method='simple_imputer.mean')
    logging.info('test')
    test = StandardCCDDataset(test, priv_ic_prob=0, unpriv_ic_prob=0,
                              method='baseline')

    df = train.get_incomplete_df(label_names=data.label_names,
        protected_attribute_names=data.protected_attribute_names)
    # logging.debug(df['priors_count=More than 3'].value_counts())
    logging.debug(train.imputed_df['priors_count=More than 3'].value_counts())
    # print(np.abs(df.corr()[data.label_names[0]]).sort_values().tail(n=5))
    df, _ = test.convert_to_dataframe()
    logging.debug(df['priors_count=More than 3'].value_counts())
    # print(np.abs(df.corr()[data.label_names[0]]).sort_values().tail(n=5))

    pmod, pmod_results = get_groupwise_performance(
        train, test, estimator, privileged=True, pos_rate=False
    )
    umod, umod_results = get_groupwise_performance(
        train, test, estimator, privileged=False, pos_rate=False
    )
    mod, mod_results = get_groupwise_performance(
        train, test, estimator, privileged=None, pos_rate=False)

    p_perf = get_model_performances(
        pmod, test, get_predictions, keep_prot=keep_prot)
    u_perf = get_model_performances(
        umod, test, get_predictions, keep_prot=keep_prot)
    m_perf = get_model_performances(
        mod, test, get_predictions, keep_prot=keep_prot)
    logging.info([np.exp(arr) for arr in mod.feature_log_prob_])
    logging.info(list(zip(mod.feature_names_in_, mod.n_categories_)))
    row = get_table_row(is_header=False, var_value=(args.dataset),
        p_perf=p_perf, u_perf=u_perf, m_perf=m_perf, variable=variable)
    print(row)