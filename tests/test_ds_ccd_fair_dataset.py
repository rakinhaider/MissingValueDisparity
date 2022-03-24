import numpy as np
from unittest import TestCase
from utils import *
from datasets.ds_ccd_fair_dataset import DSCCDFairDataset
from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric


class TestDSCCDFairDataset(TestCase):
    def test_ds_ccd_fbd(self):
        fbd = DSCCDFairDataset(n_samples=5, n_features=2, random_seed=43,
                               method='simple_imputer.mean')
        metric = BinaryLabelDatasetMetric(fbd,
              privileged_groups=fbd.privileged_groups,
              unprivileged_groups=fbd.unprivileged_groups)
        assert metric.disparate_impact() == 1.0

    def test_get_group_configs(self):
        fbd = DSCCDFairDataset(n_samples=5, n_redline=1, n_features=2,
                               random_seed=43,
                               dist={'mus': {1: np.array([10, 15]),
                                             0: np.array([5, 10])},
                                     'sigmas': [3, 3]})
        assert fbd.group_configs == [([7.5, 10], [3, 3], 0, 0),
                                     ([5, 12.5], [3, 3], 1, 0),
                                     ([7.5, 15], [3, 3], 0, 1),
                                     ([10, 12.5], [3, 3], 1, 1)]

    def test_generate_synthetic(self):
        fbd = DSCCDFairDataset(n_samples=5, n_features=2, random_seed=43)