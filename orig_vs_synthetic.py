import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import numpy as np
import pandas as pd
from utils import *
from missing_disparity.sampling.synthetic_generator import(
    synthetic, group_indices
)
from aif360.metrics import (BinaryLabelDatasetMetric as BM)
SYNTHETIC_BASE_INDEX = 100000


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


if __name__ == "__main__":
    np.random.seed(23)

    args = argparse.ArgumentParser(
            description="OrigVSSyntheticExperiement",
    )
    args.add_argument("-d", "--data", choices=["compas"], default='compas')
    args.add_argument('-m', '--model-type', choices=['gnb', 'mixednb', 'cat_nb'],
                      default='cat_nb')
    args.add_argument('-s', '--sample-mode', default=2, type=int)
    args.add_argument('--split', default=0.8, help='Train test split')
    args.add_argument('-r', '--random-seed', default=23)

    args = args.parse_args()
    sample_mode = args.sample_mode
    random_seed = args.random_seed
    dataset_orig = get_standard_dataset(args.data)
    model_type = get_estimator(args.model_type, reduce=False)
    estimator = model_type
    params = {}
    # The following line is not necessary here. Didn't remove to
    # ensure reproducibility. Inside the function we used random numbers.
    # Removing them will  result different random numbers in later experiments.
    # Similar pattern in results but different values.
    # exp_compas_orig(params, verbose=False)

    # print_table_row(is_header=True)

    for repair_rate in np.arange(0.2, 1.2, 0.2):
        logging.info(repair_rate)
        dataset_balanced = get_balanced_dataset(
            dataset_orig, sample_mode, repair_rate)
        fix_balanced_dataset(dataset_balanced)
        bm = BM(dataset_balanced, dataset_balanced.unprivileged_groups,
                dataset_balanced.privileged_groups)
        print(dataset_balanced.instance_names[-1])
        print(bm.base_rate(privileged=True))
        print(bm.base_rate(privileged=False))
        print(bm.base_rate(privileged=None))

        train_fd, test_fd = dataset_balanced.split(
            [args.split], shuffle=True, seed=args.random_seed)

        pmod, p_result = get_groupwise_performance(
            train_fd, test_fd, model_type,
            privileged=True, params=params, pos_rate=False
        )

        umod, u_result = get_groupwise_performance(
            train_fd, test_fd, model_type,
            privileged=False, params=params, pos_rate=False
        )
        mod, m_result = get_groupwise_performance(
            train_fd, test_fd, model_type,
            privileged=None, params=params, pos_rate=False
        )

        ######### Getting group-wise KL-divergence with orig distribution #########
        complete_pmod, _ = get_groupwise_performance(
            dataset_balanced, dataset_balanced, estimator, privileged=True, pos_rate=False
        )
        complete_umod, _ = get_groupwise_performance(
            dataset_balanced, dataset_balanced, estimator, privileged=False, pos_rate=False
        )

        p_pos_dist = np.array(
            [np.exp(i[0]) for i in complete_pmod.feature_log_prob_])
        u_pos_dist = np.array(
            [np.exp(i[0]) for i in complete_umod.feature_log_prob_])
        mod_pos_dist = np.array([np.exp(i[0]) for i in mod.feature_log_prob_])

        p_to_mod = KL_divergence(p_pos_dist, mod_pos_dist)
        u_to_mod = KL_divergence(u_pos_dist, mod_pos_dist)

        print('divergences')
        print(p_to_mod)
        print(u_to_mod)

        p_perf = get_model_performances(pmod, test_fd, get_predictions)
        u_perf = get_model_performances(umod, test_fd, get_predictions)
        m_perf = get_model_performances(mod, test_fd, get_predictions)
        row = get_table_row(is_header=False, p_perf=p_perf,
                            u_perf=u_perf, m_perf=m_perf)
        print(row)
