from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric
from unittest import TestCase
from datasets.ccd_fair_dataset import CCDFairDataset
import numpy as np


class TestCCDFairDataset(TestCase):
    def test_generate_missing_values(self):
        fbd = CCDFairDataset(n_samples=5, random_seed=43,
                             alpha=0.4, method='simple_imputer.mean')
        metric = BinaryLabelDatasetMetric(fbd,
            privileged_groups=fbd.privileged_groups,
            unprivileged_groups=fbd.unprivileged_groups)
        assert metric.disparate_impact() == 1.0

    def test_get_detailed_dist(self):
        dist = {'mus': {1: np.array([10, 13]),
                        0: np.array([0, 3])},
                'sigmas': [3, 3]}
        fbd = CCDFairDataset(n_samples=5, random_seed=43,
                             alpha=0.4, method='simple_imputer.mean',
                             dist=dist)
        metric = BinaryLabelDatasetMetric(fbd,
            privileged_groups=fbd.privileged_groups,
            unprivileged_groups=fbd.unprivileged_groups)

        orig = [([0, 0], [3, 3], 0, 0), ([10, 10], [3, 3], 0, 1),
                ([3, 3], [3, 3], 1, 0), ([13, 13], [3, 3], 1, 1)]
        result = fbd.get_group_configs()
        for i in range(len(orig)):
            assert result[i][2] == orig[i][2]
            assert result[i][3] == orig[i][3]
            assert np.all(result[i][0] == orig[i][0])
            assert np.all(result[i][1] == orig[i][1])

