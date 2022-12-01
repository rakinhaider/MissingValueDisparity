import numpy as np
import pandas as pd
from imputations import impute
from unittest import TestCase


class TestImputations(TestCase):
    def setUp(self):
        np.random.seed(41)
        values = [[i, i] for i in range(10)]
        s = [0] * 5 + [1] * 5
        y = [0, 1] * 5
        for i in [2, 3, 5, 6]:
            values[i][1] = None
        df = pd.DataFrame(values)
        df['sex'] = s
        df['label'] = y
        self.df = df

    def test_impute_mean(self):
        out, _ = impute(self.df, method='simple_imputer.mean')
        expected = self.df.copy(deep=True).fillna(29/6)
        assert np.alltrue(out == expected)

    def test_impute_keep_prot(self):
        df = self.df.copy(deep=True)
        df.iloc[3, 0] = 2.75
        df.iloc[4, 1] = None
        df.iloc[3, 1] = 3
        df.iloc[5, 1] = 5
        df.iloc[8, 1] = None
        expected = df.copy(deep=True)
        for i, imputed_val in zip([2, 4, 6, 8], [3, 3, 5, 7]):
            expected.iloc[i, 1] = imputed_val

        out, _ = impute(df, method='knn_imputer', keep_im_prot=True)
        assert np.alltrue(expected == out)

        expected.iloc[4, 1] = 5
        out, _ = impute(df, method='knn_imputer')
        assert np.alltrue(expected == out)

    def test_impute_keep_y(self):
        df = self.df.copy(deep=True)
        expected = self.df.copy(deep=True)
        out, _ = impute(df, method='knn_imputer', keep_y=True)
        print(out)
