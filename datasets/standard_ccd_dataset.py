import logging
import numpy as np
from scipy import sparse
from .ccd_fair_dataset import CCDFairDataset

class StandardCCDDataset(CCDFairDataset):
    def __init__(self, std_dataset, **kwargs):
        df, _ = std_dataset.convert_to_dataframe()

        instance_weight_names = std_dataset.\
            __dict__.get('instance_weight_names', None)

        self.complete_df = df

        super(StandardCCDDataset, self).__init__(
            n_samples=len(df), standard=std_dataset, complete_df=df,
            label_names=std_dataset.label_names,
            protected_attribute_names=std_dataset.protected_attribute_names,
            instance_weights_name=instance_weight_names,
            instance_names=std_dataset.instance_names,
            metadata=std_dataset.metadata,
            favorable_label=std_dataset.favorable_label,
            unfavorable_label=std_dataset.unfavorable_label, **kwargs)

    @property
    def privileged_groups(self):
        # TODO: Bad code below. No need to have standard in the class. Rethink.
        return self.standard.privileged_groups

    @property
    def unprivileged_groups(self):
        return self.standard.unprivileged_groups

    def generate_missing_matrix(self, **kwargs):
        std_dataset = self.standard
        df = self.complete_df
        protected_attribute_names = std_dataset.protected_attribute_names
        n_samples = df.shape[0]
        n_features = df.shape[1] - 1
        col_name, col_index = self._get_missing_column()
        missing_matrix = np.zeros((n_samples, n_features))
        # print(col_name, col_index)
        for r, p in zip([0, 1], [self.unpriv_ic_prob, self.priv_ic_prob]):
            selector = df[protected_attribute_names[0]] == r
            indices = [i for i, s in enumerate(selector.values) if s]
            n_missing = int(np.ceil(p*len(indices)))
            choices = np.random.choice(indices, size=n_missing, replace=False)
            missing_matrix[choices, col_index] += 1
        return sparse.csr_matrix(missing_matrix)

    def _get_missing_column(self):
        df = self.complete_df
        std_dataset = self.standard
        corr = df.corr()[std_dataset.label_names[0]]
        corr = np.abs(corr)
        corr = corr.dropna()
        corr = corr.drop(index=std_dataset.label_names[0])
        corr = corr.drop(index=std_dataset.protected_attribute_names)
        missing_column_name = corr.idxmax()
        missing_column_index = df.columns.get_loc(missing_column_name)
        return missing_column_name, missing_column_index
