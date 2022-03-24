import pandas as pd
import numpy as np
from scipy import sparse
from .ccd_fair_dataset import CCDFairDataset


class DSCCDFairDataset(CCDFairDataset):
    def __init__(self, priv_ic_prob=0.1, unpriv_ic_prob=0.4, n_redline=1,
                 **kwargs):
        """
        DSCCDFairDataset contains synthetic dataset with complete case
        disparity. `DS` indicates that the optimal models are dimensionally
        separated. That is the optimal group-wise models use different features
        for predictions, i.e., predictive feature disparity.

        The missingness disparity in rate of complete cases
        is controlled by `priv_ic_prob` and `unpriv_ic_prov`.

        :param priv_ic_prob: rate of incomplete cases in privileged group
        :param unpriv_ic_prob: rate of incomplete cases in unprivileged group
        :param kwargs: MVDFairDataset arguments.
        """
        assert n_redline <= kwargs['n_features']
        super(DSCCDFairDataset, self).__init__(
            priv_ic_prob, unpriv_ic_prob, n_redline=n_redline, **kwargs)

    def get_group_configs(self, **kwargs):
        mus_ = kwargs['dist']['mus']
        means = (mus_[0] + mus_[1]) / 2
        sigmas_ = kwargs['dist']['sigmas']
        n_redline = kwargs['n_redline']
        group_configs = []
        for cls in kwargs['classes']:
            for sensitive_group in kwargs['sensitive_groups']:
                cur_mus = []
                if sensitive_group == kwargs['privileged_group']:
                    cur_mus.extend(mus_[cls][:n_redline].tolist())
                    cur_mus.extend(means[n_redline:2*n_redline])
                    s = 1
                else:
                    cur_mus.extend(means[:n_redline])
                    cur_mus.extend(mus_[cls][n_redline:2 * n_redline].tolist())
                    s = 0
                group_configs.append((cur_mus, sigmas_, s, cls))
        return group_configs
