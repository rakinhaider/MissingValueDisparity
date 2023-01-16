from unittest import TestCase
from utils import *
from datasets import DSCCDFairDataset
from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric


class TestDSCCDFairDataset(TestCase):
    def test_ds_ccd_fbd(self):
        fbd = DSCCDFairDataset(n_samples=5, n_features=2, random_seed=43,
                               method='simple_imputer.mean', alpha=0.4)
        metric = BinaryLabelDatasetMetric(fbd,
              privileged_groups=fbd.privileged_groups,
              unprivileged_groups=fbd.unprivileged_groups)
        assert metric.disparate_impact() == 1.0

    def test_get_group_configs(self):
        fbd = DSCCDFairDataset(n_samples=5, n_redline=1, n_features=2,
                               random_seed=43, alpha=0.4,
                               dist={'mus': {1: np.array([10, 15]),
                                             0: np.array([5, 10])},
                                     'sigmas': [3, 3]})
        orig = [([7, 10], [3, 3], 0, 0), ([7, 15], [3, 3], 0, 1),
                ([5, 12], [3, 3], 1, 0), ([10, 12], [3, 3], 1, 1)]
        result = fbd.get_group_configs()
        for i in range(len(orig)):
            assert result[i][2] == orig[i][2]
            assert result[i][3] == orig[i][3]
            assert np.all(result[i][0] == orig[i][0])
            assert np.all(result[i][1] == orig[i][1])

    def test_generate_synthetic(self):
        fbd = DSCCDFairDataset(n_samples=5, n_features=2, random_seed=43,
                               alpha=0.4)