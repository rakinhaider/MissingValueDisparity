import numpy as np
import pandas as pd
from unittest import TestCase
from datasets.standard_missing_strategies import *


class TestStandardMissingStrategies(TestCase):
    def setUp(self):
        np.random.seed(43)
        self.df = pd.DataFrame(np.random.randint(0, 10, (20, 5)))
        self.df['race'] = np.random.randint(0, 2, (20, 1))
        self.df['label'] = self.df[2].values / 2

    def test_get_most_corr_column(self):
        assert get_most_corr_column(self.df, ['label'], ['race']) == (2, 2)

    def test_get_missing_matrix_by_column(self):
        print(self.df)
        mmat = get_missing_matrix_by_column(self.df, 0.2, 0.1, ['race'], 2)
        assert sum(sum(mmat)) == 4
        assert all(np.sum(mmat, axis=0) == [0, 0, 4, 0, 0, 0])
        # TODO: Should test group-wise missing percentages.

    def test_rand_col_selector(self):
        col = rand_single_col_selector(self.df.loc[1], 0.2, 0.1,
                                       ['race'], ['label'])
        assert all(col == [1])

    def test_missing_rand_col_by_sample(self):
        # Randomization doesn't maintain the missing_rates.
        # Toss a random probability, check if higher than non-missing prob.
        mmat = missing_single_col_by_sample(self.df, 0.2, 0.1, ['race'], ['label'])

    def test_rand_many_col_selector(self):
        print(rand_many_col_selector(self.df.loc[1], 0.2, 0.1,
                                     ['race'], ['label'], cutoff=0.75))
        print(rand_many_col_selector(self.df.loc[1], 0.2, 0.1,
                                     ['race'], ['label'], cutoff=0.5))
        print(rand_many_col_selector(self.df.loc[1], 0.2, 0.1,
                                     ['race'], ['label'], cutoff=0.25))