from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric
from unittest import TestCase
from datasets.synthetic.correlated_ccd_fair_dataset import CorrelatedCCDFairDataset
import numpy as np
from math import sqrt


class TestCCDFairDataset(TestCase):
    def test_generate_missing_values(self):
        fbd = CorrelatedCCDFairDataset(n_samples=5, random_seed=43,
                                       alpha=0.4, method='simple_imputer.mean')
        metric = BinaryLabelDatasetMetric(fbd,
            privileged_groups=fbd.privileged_groups,
            unprivileged_groups=fbd.unprivileged_groups)
        assert metric.disparate_impact() == 1.0

    def test_get_detailed_dist(self):
        dist = {'mus':{'x1': {0: [0, 0], 1:[10, 10]},
                    'z': [0, 2]},
                'sigmas': {'x1': {0: [3, 3], 1:[3, 3]},
                           'z': [0, 1]}
        }
        fbd = CorrelatedCCDFairDataset(n_samples=5, random_seed=43,
            alpha=0.4, method='simple_imputer.mean', dist=dist)

        orig = [([0, 0], [3, 3], 0, 0), ([10, 14], [3, sqrt(11)], 0, 1),
                ([0, 0], [3, 3], 1, 0), ([10, 14], [3, sqrt(11)], 1, 1)]
        result = fbd.get_group_configs()
        for i in range(len(orig)):
            assert result[i][2] == orig[i][2]
            assert result[i][3] == orig[i][3]
            assert np.all(result[i][0] == orig[i][0])
            assert np.all(result[i][1] == orig[i][1])

