from unittest import TestCase
from datasets.realworld.folkdatasets import FolkIncomeDataset, FolkPubCoverageDataset


class TestFolkDataset(TestCase):
    def test_folkincome(self):
        data = FolkIncomeDataset()
        assert len(data) == 196604
        assert data.__n_features__ == 10

    def test_folkpubliccoverage(self):
        data = FolkPubCoverageDataset()
        assert len(data) == 138554
        assert data.__n_features__ == 19