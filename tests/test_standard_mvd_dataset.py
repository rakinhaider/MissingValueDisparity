import numpy as np
from unittest import TestCase
from datasets.standard.standard_ccd_dataset import StandardCCDDataset
from utils import get_standard_dataset


class TestStandardMVDDataset(TestCase):
    def test__init__compas(self):
        # TODO: Check whether the imputed dataframe is equal to complete_df
        #  except missing values.
        data = get_standard_dataset('compas')
        stdccd = StandardCCDDataset(data)
        expected, _ = data.convert_to_dataframe()
        assert all(expected == stdccd.complete_df)

    def test__init__bank(self):
        data = get_standard_dataset('bank')
        stdccd = StandardCCDDataset(data)
        expected, _ = data.convert_to_dataframe()
        assert all(expected == stdccd.complete_df)

    def test__init__german(self):
        data = get_standard_dataset('german')
        stdccd = StandardCCDDataset(data)
        expected, _ = data.convert_to_dataframe()
        assert all(expected == stdccd.complete_df)

    def test_get_missing_matrix(self):
        np.random.seed(41)
        data = get_standard_dataset('compas')
        data = data.subset([i for i in range(10)])
        expected = np.zeros((10, data.features.shape[1]))
        expected[[1, 6, 7], 5] += 1
        stdccd = StandardCCDDataset(data, method='simple_imputer.mean')
        out = stdccd.generate_missing_matrix().todense()
        assert np.all(expected == out)

    def test_missing_rate(self):
        data = get_standard_dataset('german')
        # data = data.subset([i for i in range(10)])
        expected, _ = data.convert_to_dataframe()
        stdccd = StandardCCDDataset(data,
                                    priv_ic_prob=0.2, unpriv_ic_prob=0.4,
                                    method='simple_imputer.mean')
        out, _ = stdccd.convert_to_dataframe()
        incomplete = stdccd.get_incomplete_df(
            label_names=stdccd.label_names,
            protected_attribute_names=stdccd.protected_attribute_names)
        df_orig, _ = data.convert_to_dataframe()
        print(incomplete.isna().sum())
        print(df_orig['status=A14'].value_counts())
        print(stdccd.imputed_df['status=A14'].value_counts())

        # TODO: ??? Implementation unfinished.