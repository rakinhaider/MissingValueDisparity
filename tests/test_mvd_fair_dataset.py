import numpy as np
from datasets import _get_n_samples, _validate_alpha_beta
from datasets.mvd_fair_dataset import MVDFairDataset
from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric
from unittest import TestCase


class TestMVDFairDataset(TestCase):
    def test_mvd_fair_dataset(self):
        dists = [None, {'mu': np.array(
            [[[0, 2], [10, 12]],
             [[1, 3], [11, 13]]]),
            'sigma': np.ones((2, 2, 2)) * 3
        }]
        for d in dists:
            fbd = MVDFairDataset(5, dist=d, alpha=0.4, beta=1)
            metric = BinaryLabelDatasetMetric(fbd,
                privileged_groups=fbd.privileged_groups,
                unprivileged_groups=fbd.unprivileged_groups)

            assert metric.disparate_impact() == 1.0
            assert fbd.favorable_label == 1.0
            assert fbd.unfavorable_label == 0.0
            assert all(fbd.complete_df == fbd.imputed_df)

    def test_data_distributions(self):
        dist = {'mu': np.array(
            [[[0, 2], [10, 12]],
             [[1, 3], [11, 13]]]),
            'sigma': np.ones((2, 2, 2)) * 3}
        fbd = MVDFairDataset(1000000, dist=dist)
        df = fbd.complete_df
        grouped = df.groupby(['sex', 'label'])
        means = {(i[2], i[3]): i[0] for i in fbd.group_configs}
        for (s, y), grp in grouped:
            assert np.allclose(grp.mean(axis=0).values[:2], means[(s, y)],
                               atol=0.01)

    def test_get_n_samples(self):
        _validate_alpha_beta(0.5, 1, 40)
        assert _get_n_samples(40, 0.5, 1, 0, 0) == 20
        assert _get_n_samples(40, 0.5, 1, 0, 1) == 20
        assert _get_n_samples(40, 0.5, 1, 1, 0) == 20
        assert _get_n_samples(40, 0.5, 1, 1, 1) == 20

        _validate_alpha_beta(0.25, 1, 40)
        assert _get_n_samples(40, 0.25, 1, 0, 0) == 30
        assert _get_n_samples(40, 0.25, 1, 0, 1) == 10
        assert _get_n_samples(40, 0.25, 1, 1, 0) == 30
        assert _get_n_samples(40, 0.25, 1, 1, 1) == 10

        _validate_alpha_beta(0.75, 1, 40)
        assert _get_n_samples(40, 0.75, 1, 0, 0) == 10
        assert _get_n_samples(40, 0.75, 1, 0, 1) == 30
        assert _get_n_samples(40, 0.75, 1, 1, 0) == 10
        assert _get_n_samples(40, 0.75, 1, 1, 1) == 30

        _validate_alpha_beta(0.75, 2, 40)
        assert _get_n_samples(40, 0.75, 2, 0, 0) == 10
        assert _get_n_samples(40, 0.75, 2, 0, 1) == 30
        assert _get_n_samples(40, 0.75, 2, 1, 0) == 20
        assert _get_n_samples(40, 0.75, 2, 1, 1) == 60

    def test_get_group_config(self):
        dist = {'mus': {1: np.array([10, 15]),
                        0: np.array([0, 5])},
                'sigmas': [3, 3]}
        fbd = MVDFairDataset(5, 2, dist=dist, alpha=0.4)
        print(fbd.group_configs)
        res = [(np.array([-2,  3]), [3, 3], 0, 0),
               (np.array([0, 5]), [3, 3], 1, 0),
               (np.array([8, 13]), [3, 3], 0, 1),
               (np.array([10, 15]), [3, 3], 1, 1)]
        for i in range(len(res)):
            assert np.array_equal(res[i][0], fbd.group_configs[i][0])
            assert res[i][2] == fbd.group_configs[i][2]
            assert res[i][3] == fbd.group_configs[i][3]