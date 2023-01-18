from unittest import TestCase
from datasets.pima_dataset import PimaDataset


class TestStandardMVDDataset(TestCase):
    def test__init__(self):
        data = PimaDataset()
        print(data)