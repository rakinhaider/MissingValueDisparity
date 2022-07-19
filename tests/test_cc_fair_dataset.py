from aif360.metrics.binary_label_dataset_metric import BinaryLabelDatasetMetric
from unittest import TestCase
from datasets.ccd_fair_dataset import CCDFairDataset
import numpy as np


class TestCCDFairDataset(TestCase):
    def test_generate_missing_values(self):
        fbd = CCDFairDataset(n_samples=5, n_features=2, random_seed=43,
                             alpha=0.4)
        print(fbd.get_incomplete_df())
        print(fbd.features)
        fbd = CCDFairDataset(n_samples=5, n_features=2, random_seed=43,
                             alpha=0.4, method='simple_imputer.mean')
        metric = BinaryLabelDatasetMetric(fbd,
            privileged_groups=fbd.privileged_groups,
            unprivileged_groups=fbd.unprivileged_groups)
        assert metric.disparate_impact() == 1.0