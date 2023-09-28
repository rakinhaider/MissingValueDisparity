from unittest import TestCase
from datasets.realworld.pima_dataset import PimaDataset


class TestPIMADataset(TestCase):
    def test__init__(self):
        data = PimaDataset()
        assert len(data) == 768