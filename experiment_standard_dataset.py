import os
import pickle
import warnings

warnings.simplefilter(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
from datasets.standard_ccd_dataset import StandardCCDDataset


def save_models(args, mod_to_name_map, fold_id):
    out_dir = 'outputs/standard/{}/models/{}'.format(
        args.dataset, args.estimator)
    os.makedirs(out_dir, exist_ok=True)
    for model in mod_to_name_map:
        mod_name = mod_to_name_map[model]
        model_fname = '{:s}/{:s}_{:s}_{:s}_{:s}_{:.2f}_{:.2f}_{:s}_{:d}_' \
                      '{:d}_{:d}_{}.pkl'.format(
            out_dir, mod_name, args.dataset, args.estimator,
            METHOD_SHORTS[args.method], args.unpriv_ic_prob, args.priv_ic_prob,
            args.calibrate if args.calibrate else 'None', args.strategy,
            1 if args.reduce else 0, 1 if args.xvalid else 0, fold_id
        )
        pickle.dump(model, open(model_fname, 'wb'))


def save_probas(args, proba, fold_id):
    out_dir = 'outputs/standard/{}/proba/{}'.format(args.dataset, args.estimator)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(proba, columns=[0, 1]).to_csv(
        '{:s}/{:s}_{:s}_{:s}_{:.2f}_{:.2f}_{:s}_{:d}_{:d}_{:d}_{}.tsv'.format(
            out_dir, args.dataset, args.estimator, METHOD_SHORTS[args.method],
            args.unpriv_ic_prob, args.priv_ic_prob,
            args.calibrate if args.calibrate else 'None', args.strategy,
            1 if args.reduce else 0, 1 if args.xvalid else 0, fold_id
        ), sep='\t'
    )


def experiment(std_train, std_test, args, fold_id=None, **kwargs):
    train = StandardCCDDataset(std_train, priv_ic_prob=args.priv_ic_prob,
                               unpriv_ic_prob=args.unpriv_ic_prob,
                               method=args.method, strategy=args.strategy,
                               missing_column_name=kwargs['col'])
    incomplete_df = train.get_incomplete_df(
        protected_attribute_names=train.protected_attribute_names,
        label_names=train.label_names, instance_names=std_train.instance_names)
    logging.info(incomplete_df.describe().loc['count'])
    test = StandardCCDDataset(std_test, priv_ic_prob=0, unpriv_ic_prob=0,
                              method='baseline')

    keep_features = 'all'
    cali_kwargs = {'calibrate': args.calibrate,
                   'calibrate_cv': args.calibrate_cv}
    mod, m_perf = get_groupwise_performance(
        estimator, train, test, privileged=None,
        keep_features=keep_features, **cali_kwargs
    )

    # Keep prot is used from above.
    test_x, test_y = get_xy(test, keep_protected=keep_prot,
                            keep_features=keep_features)
    proba = mod.predict_proba(test_x)
    save_probas(args, proba, fold_id)
    save_models(args, {mod: 'mod'}, fold_id)

    ######### Getting group-wise KL-divergence with test distribution #########
    complete_pmod, _ = get_groupwise_performance(
        estimator, std_train, std_train, privileged=True, **cali_kwargs)
    complete_umod, _ = get_groupwise_performance(
        estimator, std_train, std_train, privileged=False, **cali_kwargs)
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
        var_val = (args.dataset, col, METHOD_SHORTS[args.method],
                   args.unpriv_ic_prob, args.priv_ic_prob, fold_id)
    else:
        var_val = (args.dataset, METHOD_SHORTS[args.method],
                   args.unpriv_ic_prob, args.priv_ic_prob, fold_id)
    row = get_table_row(is_header=False, p_perf=m_perf, u_perf=m_perf,
                        m_perf=m_perf, variable=variable,
                        var_value=var_val)
    print(row, flush=True)
    logging.StreamHandler().flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-dt',
                        choices=['compas', 'bank', 'german', 'adult', 'pima'])
    parser.add_argument('--priv-ic-prob', '-pic', default=0.1, type=float)
    parser.add_argument('--unpriv-ic-prob', '-upic', default=0.4, type=float)

    parser.add_argument('--method', default='simple_imputer.mean',
                        choices=['baseline', 'drop', 'simple_imputer.mean',
                                 'simple_imputer.mode', 'iterative_imputer.mice',
                                 'iterative_imputer.missForest', 'knn_imputer'])
    parser.add_argument('--reduce', '-r', action='store_true', default=False)
    parser.add_argument('--calibrate', '-c', default=None,
                        choices=['sigmoid', 'isotonic'])
    parser.add_argument('--calibrate-cv', '-ccv', type=int, default=10)
    parser.add_argument('--xvalid', '-x', default=False, action='store_true')
    parser.add_argument('--strategy', '-s', type=int, default=0)

    parser.add_argument('--header-only', default=False, action='store_true')

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
        variable = ('dataset', 'col', 'method', 'upic', 'pic', 'fold')
    else:
        variable = ('dataset', 'method', 'upic', 'pic', 'fold')
    if args.print_header or args.header_only:
        print(get_table_row(is_header=True, variable=variable))
        if args.header_only:
            exit()

    np.random.seed(args.random_seed)

    data = get_standard_dataset(args.dataset)

    if args.xvalid:
        std_train = data.copy(deepcopy=True)
        std_test = None
    else:
        std_train, std_test = data.split([0.8], shuffle=True,
                                         seed=args.random_seed)

    if args.strategy == 1:
        columns = data.feature_names
    else:
        columns = [None]

    for col in columns:
        if col in data.protected_attribute_names + data.label_names:
            continue
        if args.xvalid:
            x, y = get_xy(std_train, keep_protected=keep_prot)
            kf = KFold(n_splits=5)
            for i, (train_indices, test_indices) in enumerate(kf.split(x, y)):
                exp_train = std_train.subset(train_indices)
                exp_test = std_train.subset(test_indices)
                experiment(exp_train, exp_test, args, i, keep_prot=keep_prot,
                           col=col)
        else:
            experiment(std_train, std_test, args, 'n', keep_prot=keep_prot,
                       col=col)