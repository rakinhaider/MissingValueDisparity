import numpy as np
from datasets.synthetic.mvd_fair_dataset import FairDataset
from unittest import TestCase


class TestFairDataset(TestCase):
    def test__get_default_dist(self):
        out = FairDataset._get_default_dist()
        dist = {
            'mu': np.array([[[10., 10.], [10., 10.]],
                            [[10., 10.], [10., 10.]]]),
            'sigma': np.array([[[3., 3.], [3., 3.]],
                               [[3., 3.], [3., 3.]]])}

        for key in dist:
            assert np.all(out[key] == dist[key])
