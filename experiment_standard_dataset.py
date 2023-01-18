import logging
import os
import pickle
import warnings

import pandas as pd

warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
from datasets.standard_ccd_dataset import StandardCCDDataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['compas', 'bank', 'german', 'adult', 'pima'])
    parser.add_argument('--print-header', default=False, action='store_true')
    parser.add_argument('--method', default='simple_imputer.mean',
                        choices=['baseline', 'drop', 'simple_imputer.mean',
                                 'simple_imputer.mode', 'iterative_imputer.mice',
                                 'iterative_imputer.missForest', 'knn_imputer'])
    parser.add_argument('--estimator', '-e', default='cat_nb',
                        choices=['cat_nb', 'nb', 'lr', 'svm', 'pr', 'dt'])
    parser.add_argument('--reduce', '-r', action='store_true', default=False)
    parser.add_argument('--calibrate', '-c', default=None,
                        choices=['sigmoid', 'isotonic'])
    parser.add_argument('--calibrate-cv', '-ccv', type=int, default=10)
    parser.add_argument('--strategy', '-s', type=int, default=0)

    parser.add_argument('--header-only', default=False, action='store_true')

    parser.add_argument('--priv-ic-prob', '-pic', default=0.1, type=float)
    parser.add_argument('--unpriv-ic-prob', '-upic', default=0.4, type=float)

    parser.add_argument('--random-seed', default=41, type=int)
    parser.add_argument('--log-level', '-ll', default='ERROR')

    args = parser.parse_args()
    args.reduce = False
    method = args.method
    estimator = get_estimator(args.estimator, False)
    keep_prot = args.reduce or (args.estimator == 'pr')
    # keep_prot = False
    cali_kwargs = {'calibrate': args.calibrate,
                   'calibrate_cv': args.calibrate_cv}

    LOG_FORMAT = '%(asctime)s - %(module)s - %(lineno)d - %(levelname)s \n %(message)s'
    level = logging.getLevelName(args.log_level)
    logging.basicConfig(level=level, format=LOG_FORMAT)

    if args.strategy == 1:
        variable = ('dataset', 'col', 'upic')
    else:
        variable = ('dataset', 'upic')
    if args.print_header or args.header_only:
        print(get_table_row(is_header=True, variable=variable))
        if args.header_only:
            exit()

    np.random.seed(args.random_seed)

    data = get_standard_dataset(args.dataset)
    std_train, std_test = data.split([0.8], shuffle=True, seed=args.random_seed)

    if args.strategy == 1:
        columns = data.feature_names
    else:
        columns = [None]

    for col in columns:
        if col in data.protected_attribute_names + data.label_names:
            continue

        train = StandardCCDDataset(std_train, priv_ic_prob=args.priv_ic_prob,
                                   unpriv_ic_prob=args.unpriv_ic_prob,
                                   method=args.method, strategy=args.strategy,
                                   missing_column_name=col)
        incomplete_df = train.get_incomplete_df(
            protected_attribute_names=train.protected_attribute_names,
            label_names=train.label_names, instance_names=train.instance_names)
        logging.info(incomplete_df.describe().loc['count'])
        test = StandardCCDDataset(std_test, priv_ic_prob=0, unpriv_ic_prob=0,
                                  method=args.method)

        keep_features = 'all'
        pmod, pmod_results = get_groupwise_performance(
            train, test, estimator, privileged=True, pos_rate=False,
            keep_features=keep_features, **cali_kwargs
        )
        umod, umod_results = get_groupwise_performance(
            train, test, estimator, privileged=False, pos_rate=False,
            keep_features=keep_features, **cali_kwargs
        )
        mod, mod_results = get_groupwise_performance(
            train, test, estimator, privileged=None, pos_rate=False,
            keep_features=keep_features, **cali_kwargs
        )
        p_perf = get_model_performances(pmod, test, get_predictions,
            keep_prot=keep_prot, keep_features=keep_features, **cali_kwargs)
        u_perf = get_model_performances(umod, test, get_predictions,
            keep_prot=keep_prot, keep_features=keep_features, **cali_kwargs)
        m_perf = get_model_performances(mod, test, get_predictions,
            keep_prot=keep_prot, keep_features=keep_features, **cali_kwargs)

        test_x, test_y = get_xy(test, keep_protected=keep_prot,
                                keep_features=keep_features)
        proba = mod.predict_proba(test_x)
        out_dir = 'outputs/standard/{}/'.format(args.dataset)
        proba_dir = '{:s}/proba/{}'.format(out_dir, args.estimator)
        model_dir = '{:s}/models/{}'.format(out_dir, args.estimator)
        os.makedirs(proba_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        pd.DataFrame(proba, columns=[0, 1]).to_csv(
            '{:s}/{:s}_{:s}_{:.2f}_{:.2f}.tsv'.format(
                proba_dir, args.dataset, args.estimator,
                args.unpriv_ic_prob, args.priv_ic_prob),
            sep='\t'
        )
        for model, mod_name in zip([pmod, umod, mod], ['pmod', 'umod', 'mod']):
            model_fname = '{:s}/{:s}_{:.2f}_{:.2f}.pkl'.format(
                model_dir, mod_name, args.unpriv_ic_prob, args.priv_ic_prob)
            pickle.dump(model, open(model_fname, 'wb'))

        ######### Getting group-wise KL-divergence with test distribution #########
        complete_pmod, _ = get_groupwise_performance(
            std_train, std_train, estimator, privileged=True,
            pos_rate=False, **cali_kwargs
        )
        complete_umod, _ = get_groupwise_performance(
            std_train, std_train, estimator, privileged=False,
            pos_rate=False, **cali_kwargs
        )
        # logging.info(mod.feature_log_prob_)
        # logging.info(complete_pmod.feature_log_prob_)
        # logging.info(complete_umod.feature_log_prob_)

        # p_pos_dist = np.array([np.exp(i[1]) for i in complete_pmod.feature_log_prob_])
        # u_pos_dist = np.array([np.exp(i[1]) for i in complete_umod.feature_log_prob_])
        # mod_pos_dist = np.array([np.exp(i[1]) for i in mod.feature_log_prob_])

        # p_to_mod = KL_divergence(p_pos_dist, mod_pos_dist)
        # u_to_mod = KL_divergence(u_pos_dist, mod_pos_dist)

        # logging.info(p_to_mod)
        # logging.info(u_to_mod)
        if args.strategy == 1:
            var_val = (args.dataset, col, args.unpriv_ic_prob)
        else:
            var_val = (args.dataset, args.unpriv_ic_prob)
        row = get_table_row(is_header=False, p_perf=m_perf, u_perf=m_perf,
                            m_perf=m_perf, variable=variable,
                            var_value=var_val)
        print(row, flush=True)
        logging.StreamHandler().flush()
