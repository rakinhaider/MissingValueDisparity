from unittest import TestCase
from datasets.realworld.heart_dataset import HeartDataset


class TestHeartDataset(TestCase):
    def test__init__(self):
        data = HeartDataset()
        assert len(data) == 70000
        assert data.__n_features__ == 12