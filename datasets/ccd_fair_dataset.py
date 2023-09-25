import logging
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
        self.priv_ic_prob = priv_ic_prob
        self.unpriv_ic_prob = unpriv_ic_prob
        super(CCDFairDataset, self).__init__(**kwargs)

    def generate_missing_matrix(self, **kwargs):
        rng = kwargs['rng']
        n_samples = self.complete_df.shape[0]
        n_features = self.complete_df.shape[1] - 2
        label_name = kwargs.get('label_name', 'label')
        missing_col = kwargs.get('n_redline', 1)
        protected_attribute_names = kwargs. \
            get('protected_attribute_names', ['sex'])
        selector = [protected_attribute_names[0]] + [label_name]
        r = sparse.csr_matrix(np.zeros((n_samples, n_features)))
        for grp, count in self.complete_df[selector].value_counts().sort_index(ascending=False).items():
            if grp[0] == 1:
                n_incomplete = int(count * self.priv_ic_prob)
            else:
                n_incomplete = int(count * self.unpriv_ic_prob)
            selection = (self.complete_df[selector] == grp)
            indices = selection[selection.all(axis=1)].index
            incomplete_cases = rng.choice(
                list(indices), size=n_incomplete, replace=False)
            r[incomplete_cases, missing_col] = 1
        return r

    def get_detailed_dist(self, dist, **kwargs):
        if dist is None:
            return None
        classes = kwargs['classes']
        sensitive_groups = kwargs['sensitive_groups']
        n_class = len(classes)
        n_group = len(sensitive_groups)
        n_features = len(dist['mus'][1])
        formatted_mus = [[None for _ in range(n_class)]for _ in range(n_group)]
        for sensitive_group in sensitive_groups:
            for cls in classes:
                if sensitive_group == kwargs['privileged_group']:
                    s = 1
                else:
                    s = 0
                cur_mus = [dist['mus'][cls][s] for _ in range(n_features)]
                formatted_mus[s][cls] = cur_mus
        formatted_mus = np.array(formatted_mus)
        formatted_sigmas = np.zeros((n_group, n_class, n_features))
        # TODO: Fix for different class-wise sigmas
        for i in range(n_group):
            formatted_sigmas[i, :, :] += dist['sigmas'][i]
        formatted_dist = {'mu': formatted_mus, 'sigma': formatted_sigmas}
        logging.info(formatted_dist)
        return formatted_dist