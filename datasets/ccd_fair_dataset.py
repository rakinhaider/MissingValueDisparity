import pandas as pd
import numpy as np
from scipy import sparse
from .mvd_fair_dataset import MVDFairDataset


class CCDFairDataset(MVDFairDataset):
    def __init__(self, priv_ic_prob=0.1, unpriv_ic_prob=0.4, **kwargs):
        """
        CCDFairDataset contains synthetic dataset with complete case disparity.
        The disparity in rate of complete cases are controlled by `priv_ic_prob`
        and `unpriv_ic_prov`.
        :param priv_ic_prob: rate of incomplete cases in privileged group
        :param unpriv_ic_prob: rate of incomplete cases in unprivileged group
        :param kwargs: MVDFairDataset arguments.
        """
        self.priv_cc_prob = priv_ic_prob
        self.unpriv_cc_prob = unpriv_ic_prob
        super(CCDFairDataset, self).__init__(**kwargs)

    def generate_missing_matrix(self, **kwargs):
        n_samples = self.complete_df.shape[0]
        n_features = self.complete_df.shape[1] - 2
        # missing_col = np.random.randint(0, n_features)
        missing_col = kwargs.get('n_redline', 1)
        protected_attribute_names = kwargs. \
            get('protected_attribute_names', ['sex'])
        s_name = protected_attribute_names[0]
        r = sparse.csr_matrix(np.zeros((n_samples, n_features)))
        for grp, count in self.complete_df[s_name].value_counts().iteritems():
            if grp == 1:
                n_incomplete = int(count * self.priv_cc_prob)
            else:
                n_incomplete = int(count * self.unpriv_cc_prob)

            indices = self.complete_df[self.complete_df[s_name] == grp].index
            incomplete_cases = np.random.choice(
                list(indices), size=n_incomplete, replace=False)
            r[incomplete_cases, missing_col] = 1

        return r
