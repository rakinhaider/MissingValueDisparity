import logging

import numpy as np
import math
from .ccd_fair_dataset import CCDFairDataset


class CorrelatedCCDFairDataset(CCDFairDataset):
    def __init__(self, priv_ic_prob=0.1, unpriv_ic_prob=0.4, **kwargs):
        """
        CorreleatedCCDFairDataset contains synthetic dataset with complete case
        disparity. `Correlated` indicates that the features are correlate to
        each other. r$x_1$ and r$x_2$ are observer variables and $z$ is
        a latent variable. r$z$ is dependent on s. $x_2$ follows a linear
        combination of $x_1$ and $z$. That is the optimal group-wise models
        are different. We assume there are no other features available.

        The missingness disparity in rate of complete cases
        is controlled by `priv_ic_prob` and `unpriv_ic_prov`.

        :param priv_ic_prob: rate of incomplete cases in privileged group
        :param unpriv_ic_prob: rate of incomplete cases in unprivileged group
        :param kwargs: MVDFairDataset arguments.
        """
        super(CorrelatedCCDFairDataset, self).__init__(
            priv_ic_prob, unpriv_ic_prob, **kwargs)

    def get_detailed_dist(self, dist, **kwargs):
        """
        Format of dist
        dist = {'mus':{'x1': {0: [0, 0], 1:[10, 10]}, 'z': [0, 2]},
               'sigmas': {'x1':{0: [3, 3], 1:[3, 3]}, 'z': [0, 1]}
        }
        Linear combination:
            r'x_2 = x_1 + 2 z'
        :param dist:
        :param kwargs:
        :return:
        """
        if dist is None:
            return None
        classes = kwargs['classes']
        sensitive_groups = kwargs['sensitive_groups']
        n_class = len(classes)
        n_group = len(sensitive_groups)
        formatted_mus = [[None for _ in range(n_class)]for _ in range(n_group)]
        for sensitive_group in sensitive_groups:
            for cls in classes:
                if sensitive_group == kwargs['privileged_group']:
                    s = 1
                else:
                    s = 0
                mu_x1 = dist['mus']['x1'][cls][s]
                mu_z = dist['mus']['z'][cls]
                # TODO: Vary weights of linear combination
                cur_mus = [mu_x1, mu_x1 + 2 * mu_z]
                formatted_mus[s][cls] = cur_mus
        formatted_mus = np.array(formatted_mus)
        formatted_sigmas = np.zeros((n_group, n_class, 2))
        # TODO: Fix for different class-wise sigmas
        # TODO: Fix weights of linear combination
        for i in range(n_group):
            for j in range(n_class):
                x1_sigma = dist['sigmas']['x1'][j][i]
                z_sigma = dist['sigmas']['z'][j]
                x2_sigma = math.sqrt(x1_sigma ** 2 + 2 * z_sigma ** 2)
                formatted_sigmas[i, j, :] = [x1_sigma, x2_sigma]

        formatted_dist = {'mu': formatted_mus, 'sigma': formatted_sigmas}
        logging.info(formatted_dist)
        return formatted_dist