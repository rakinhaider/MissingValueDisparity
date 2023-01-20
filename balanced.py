import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from utils import *
from datasets.standard_ccd_dataset import StandardCCDDataset


def get_indices(dataset, condition=None, label=None):
    cond_vec = condition_vec(dataset.features,
                             dataset.feature_names, condition)

    if label is not None:
        truth_values = dataset.labels == label
        truth_values = truth_values.reshape(-1)
        cond_vec = np.logical_and(cond_vec, truth_values)

    return [i for i, val in enumerate(cond_vec) if val]


def get_pos_neg_by_group(dataset, group):
    pos = compute_num_pos_neg(dataset.features, dataset.labels,
                              dataset.instance_weights,
                              dataset.feature_names,
                              dataset.favorable_label, group)

    neg = compute_num_pos_neg(dataset.features, dataset.labels,
                              dataset.instance_weights,
                              dataset.feature_names,
                              dataset.unfavorable_label, group)
    return pos, neg


def get_samples_by_group(dataset, n_samples, group, label):
    indices = get_indices(dataset, group, label)
    indices = np.random.choice(indices, n_samples, replace=False)
    return indices


def sample_fd_indices(dataset, alpha=0.5, beta=1):
    # First find n_unprivileged
    # Then check number of samples to take from each group
    # Then sample and merge them together.
    p_group, u_group = get_group_dicts(dataset)
    p_pos, p_neg = get_pos_neg_by_group(dataset, p_group)
    u_pos, u_neg = get_pos_neg_by_group(dataset, u_group)

    n_unpriv = min(u_pos/alpha, u_neg/(1 - alpha),
                   p_pos/(alpha * beta), p_neg/((1-alpha)*beta)
    )
    n_priv = beta * n_unpriv

    f_label = dataset.favorable_label
    uf_label = dataset.unfavorable_label

    temp = [(alpha * n_unpriv, u_group, f_label),
            ((1 - alpha) * n_unpriv, u_group, uf_label),
            (alpha * n_priv, p_group, f_label),
            ((1 - alpha) * n_priv, p_group, uf_label)]

    indices = []
    for n, g, f in temp:
        # print(n)
        sample_indices = get_samples_by_group(dataset, round(n), g, f)
        indices.extend(sample_indices)

    return indices


def get_real_fd(dataset, alpha=0.5, beta=1):
    indices = sample_fd_indices(dataset, alpha, beta)
    fair_real_dataset = dataset.subset(indices)
    u_group, p_group = get_group_dicts(dataset)
    dataset_metric = BM(fair_real_dataset, u_group, p_group)
    assert abs(dataset_metric.base_rate(privileged=True) - alpha) < 1e-3
    assert abs(dataset_metric.base_rate(privileged=False) - alpha) < 1e-3
    assert abs(dataset_metric.base_rate(privileged=None) - alpha) < 1e-3
    fd = FairDataset(2, 1, 1)
    fd.update_from_dataset(fair_real_dataset)
    return fd


def get_balanced_dataset(dataset, sample_mode, sampling_strategy):
    f_label = dataset.favorable_label
    uf_label = dataset.unfavorable_label
    p_group, u_group = dataset.privileged_groups, dataset.unprivileged_groups

    dataset_metric = BM(dataset, u_group, p_group)

    base_rate_p = dataset_metric.base_rate(privileged=True)
    base_rate_u = dataset_metric.base_rate(privileged=False)

    dataset = synthetic(dataset, u_group,
                        base_rate_p, base_rate_u,
                        f_label, uf_label,
                        None, sample_mode, sampling_strategy)

    return dataset


def fix_balanced_dataset(dataset):
    n = len(dataset.features)
    # Fixing balanced dataset attributes
    # dataset.instance_names = [str(i) for i in range(n)]
    instance_names = dataset.instance_names.copy()
    new_samples = dataset.features.shape[0] - len(instance_names)
    instance_names += [str(SYNTHETIC_BASE_INDEX + i) for i in range(new_samples)]
    dataset_balanced.instance_names = instance_names
    dataset.scores = np.ones_like(dataset.labels)
    dataset.scores -= dataset.labels
    li = len(dataset.instance_names)
    lf = len(dataset.features)
    ls = len(dataset.scores)
    assert li == lf and lf == ls


def stratify_data(data):
    df, _ = data.convert_to_dataframe()
    grouped = df.groupby(data.protected_attribute_names + data.label_names)
    min_group = min([len(grp_df) for _, grp_df in grouped])
    selected_indices = []
    for grp, grp_df in grouped:
        choice = np.random.choice(list(grp_df.index), min_group, replace=False)
        selected_indices.extend(choice)

    selected_indices = [data.instance_names.index(i) for i in selected_indices]
    return data.subset(selected_indices)


def introduce_missing_values(train, test, args):
    train = StandardCCDDataset(train, priv_ic_prob=args.pic,
                               unpriv_ic_prob=args.uic, strategy=3,
                               method=args.method)
    incomplete_df = train.get_incomplete_df(
        protected_attribute_names=train.protected_attribute_names,
        label_names=train.label_names, instance_names=train.instance_names)
    logging.info(incomplete_df.describe().loc['count'])

    test = StandardCCDDataset(test, priv_ic_prob=0, unpriv_ic_prob=0,
                              method='baseline')
    return train, test


if __name__ == "__main__":
    np.random.seed(23)

    args = argparse.ArgumentParser(
            description="BalancedExperiement",
    )
    args.add_argument("-d", "--data", choices=["compas", "pima"],
                      default='compas')
    args.add_argument('-m', '--model-type', default='cat_nb',
                      choices=['nb', 'mixednb', 'cat_nb', 'lr'])
    args.add_argument('--split', default=0.8, help='Train test split')
    args.add_argument('--method', default='simple_imputer.mean',
                      help='Imputation method')
    args.add_argument('--uic', default=0.3, type=float,
                      help='Unprivileged missing rate')
    args.add_argument('--pic', default=0.1, type=float,
                      help='Privileged missing rate')
    args.add_argument('-r', '--random-seed', default=47)
    args.add_argument('-ll', '--log-level', default=logging.ERROR)

    args = args.parse_args()
    random_seed = args.random_seed
    dataset_orig = get_standard_dataset(args.data)
    model_type = get_estimator(args.model_type, reduce=False)
    estimator = model_type
    params = {}

    np.random.seed(random_seed)
    logging.basicConfig(level=args.log_level)

    # Orig
    data = get_standard_dataset(args.data)
    # Stratify
    data = stratify_data(data)
    # Train Test Split
    train, test = data.split([0.8], shuffle=True)
    # Missing in train
    train, test = introduce_missing_values(train, test, args)

    variable = ('data', 'uic', 'pic')
    keep_prot = False

    pmod, p_perf = get_groupwise_performance(
        estimator, train, test, privileged=True)
    umod, u_perf = get_groupwise_performance(
        estimator, train, test, privileged=False)
    mod, m_perf = get_groupwise_performance(
        estimator, train, test, privileged=None)

    test_x, test_y = get_xy(test, keep_protected=False)
    proba = mod.predict_proba(test_x)
    out_dir = 'outputs/standard/{}/balanced/'.format(args.data)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(proba, columns=[0, 1]).to_csv(
        out_dir + 'proba_{:s}_{:s}_{:.2f}.tsv'.format(
            args.data, args.model_type, args.uic),
        sep='\t'
    )

    row = get_table_row(is_header=False, p_perf=p_perf, u_perf=u_perf,
                        m_perf=m_perf, variable=variable,
                        var_value=(args.data, args.uic, args.pic))
    print(row, flush=True)
    logging.StreamHandler().flush()
