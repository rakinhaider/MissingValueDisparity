from unittest import TestCase
from aif360.datasets.compas_dataset import CompasDataset
from utils import *

class TestUtils(TestCase):
    def test_get_privileged_group(self):
        data = CompasDataset()
        # print(get_privileged_groups(data))

    def test_get_standard_dataset_compas(self):
        data = get_standard_dataset('compas')
        assert data.unprivileged_groups == [{'race': 0}]
        assert data.privileged_groups == [{'race': 1}]
        df = data.metadata['params']['df']
        assert len(df[df['race'] == 0]) == 3175
        assert len(df[df['race'] == 1]) == 2103

    def test_get_standard_dataset_german(self):
        data = get_standard_dataset('german')
        assert data.privileged_groups == [{'age': 1}]
        assert data.unprivileged_groups == [{'age': 0}]
        df = data.metadata['params']['df']
        assert len(df[df['age'] == 0]) == 190
        assert len(df[df['age'] == 1]) == 810

    def test_get_standard_dataset_bank(self):
        data = get_standard_dataset('bank')
        assert data.privileged_groups == [{'age': 1}]
        assert data.unprivileged_groups == [{'age': 0}]
        df = data.metadata['params']['df']
        assert len(df[df['age'] == 0]) == 864
        assert len(df[df['age'] == 1]) == 29624

    def test_get_standarad_dataset_adult(self):
        data = get_standard_dataset('adult')
        print(data.__dict__)

    def test_get_samples_by_group(self):
        data = get_standard_dataset('compas')
        for privileged, count, group_id in [(False, 3175, 0), (True, 2103, 1)]:
            group = get_samples_by_group(data, privileged)
            assert len(group.features) == count

            df, _ = group.convert_to_dataframe()
            counts = df['race'].value_counts()
            assert len(counts) == 1
            assert counts.index[0] == group_id

    def test_get_table_row(self):
        cols = ["AC_p", "AC_u", "SR_p", "SR_u", "FPR_p", "FPR_u"]
        perf = {key: 0.5 for key in cols}
        expecteds = [
            "alpha	 & 	$AC_p$	 & 	$AC_u$	 & 	$SR_p$	 & 	$SR_u$	 & 	$FPR_p$	 & 	$FPR_u$\\\\",
            "alpha	 & 	method	 & 	$AC_p$	 & 	$AC_u$	 & 	$SR_p$	 & 	$SR_u$	 & 	$FPR_p$	 & 	$FPR_u$\\\\",
            "0.25	 & 	baseline	 & 	00.5	 & 	00.5	 & 	00.5	 & 	00.5	 & 	00.5	 & 	00.5	 & 	00.5	 & 	00.5\\\\",
        ]

        for i, (header, var_value, variable) in enumerate([
            (True, 0.25, "alpha"),
            (True, (0.25, "baseline"), ("alpha", "method")),
            (False, (0.25, "baseline"), ("alpha", "method")),
        ]):
            out = get_table_row(is_header=header, var_value=var_value, m_perf=perf, variable=variable)
            assert out == expecteds[i]

    def test_kl_divergence(self):
        p = np.array([[0.1, 0.4, 0.5], [0.1, 0.4, 0.5]])
        q = np.array([[0.8, 0.15, 0.05], [0.8, 0.15, 0.05]])
        output = KL_divergence(p, q)
        assert np.isclose(output, -1.926979047*2)

    def test_train_model(self):
        model = train_model(GaussianNB, BankDataset(), {}, keep_prot=False,
                            calibrate='sigmoid', calibrate_cv=10)

        print(model)
