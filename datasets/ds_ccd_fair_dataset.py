import numpy as np
from .ccd_fair_dataset import CCDFairDataset


class DSCCDFairDataset(CCDFairDataset):
    def __init__(self, priv_ic_prob=0.1, unpriv_ic_prob=0.4,
                 n_redline=1, **kwargs):
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
            priv_ic_prob, unpriv_ic_prob, n_redline=n_redline,
            **kwargs)

    def get_detailed_dist(self, dist, **kwargs):
        if dist is None:
            return None
        classes = kwargs['classes']
        sensitive_groups = kwargs['sensitive_groups']
        n_class = len(classes)
        n_group = len(sensitive_groups)
        n_redline = kwargs['n_redline']
        n_features = kwargs['n_features']
        alpha = kwargs['alpha']
        means = (dist['mus'][0] * (1 - alpha) + dist['mus'][1] * alpha)
        formatted_mus = [[None for _ in range(n_class)]for _ in range(n_group)]
        for sensitive_group in sensitive_groups:
            for cls in classes:
                cur_mus = []
                if sensitive_group == kwargs['privileged_group']:
                    cur_mus.extend(dist['mus'][cls][:n_redline].tolist())
                    cur_mus.extend(means[n_redline:2 * n_redline])
                    cur_mus.extend(dist['mus'][cls][2*n_redline:].tolist())
                    s = 1
                else:
                    cur_mus.extend(means[:n_redline])
                    cur_mus.extend(dist['mus'][cls][n_redline:2 * n_redline].tolist())
                    cur_mus.extend(dist['mus'][cls][2 * n_redline:].tolist())
                    s = 0
                formatted_mus[s][cls] = cur_mus
        formatted_mus = np.array(formatted_mus)
        formatted_sigmas = np.zeros((n_group, n_class, n_features))
        # TODO: Fix for different class-wise sigmas
        for i in range(n_group):
            formatted_sigmas[i, :, :] += dist['sigmas'][i]
        formatted_dist = {'mu': formatted_mus, 'sigma': formatted_sigmas}
        return formatted_dist
