import warnings
from aif360.metrics import (
    BinaryLabelDatasetMetric,
    ClassificationMetric
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
from inherent_bias.fair_dataset import FairDataset, default_mappings
from inherent_bias.ds_fair_dataset import DSFairDataset
from scipy.special import erf
from math import sqrt
import scipy.stats as stats
import matplotlib.pyplot as plt
from fairml import ExponentiatedGradientReduction, PrejudiceRemover

warnings.simplefilter(action='ignore')


def get_single_prot_default_map():
    metadata = default_mappings.copy()
    metadata['protected_attribute_maps'] = [{1.0: 'Male', 0.0: 'Female'}]
    return metadata


def get_dataset_metrics(fd_train,
                        verbose=False):
    unprivileged_groups = fd_train.unprivileged_groups
    privileged_groups = fd_train.privileged_groups
    metrics = BinaryLabelDatasetMetric(fd_train,
                                       unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)

    if verbose:
        print('Mean Difference:', metrics.mean_difference())
        print('Dataset Base Rate', metrics.base_rate(privileged=None))
        print('Privileged Base Rate', metrics.base_rate(privileged=True))
        print('Protected Base Rate', metrics.base_rate(privileged=False))
        print('Disparate Impact:', metrics.disparate_impact())
        # print('Confusion Matrix:', metrics.binary_confusion_matrix())

    return metrics.mean_difference(), metrics.disparate_impact()



def plot_lr_boundary(clf, plt, label):
    # Retrieve the model parameters.
    b = clf.intercept_[0]
    w1, w2 = clf.coef_.T
    # Calculate the intercept and gradient of the decision boundary.
    c = -b / w2
    m = -w1 / w2

    # Plot the data and the classification with the decision boundary.
    xmin, xmax = -10, 20
    ymin, ymax = -10, 25
    xd = np.array([xmin, xmax])
    yd = m * xd + c
    plt.plot(xd, yd, lw=1, label=label)
    if label:
        plt.legend()


def get_model_properties(model):
    if isinstance(model, DecisionTreeClassifier):
        return model.get_depth()
    elif isinstance(model, LogisticRegression):
        return model.coef_
    elif isinstance(model, GaussianNB):
        return model.theta_, np.sqrt(model.sigma_)






def di_theta(delta_mu_c, delta_mu_a, sigma_u, sigma_p):
    num = 2 
    num += erf((delta_mu_c - delta_mu_a)/(sqrt(2)*sigma_u))
    num -= erf((delta_mu_c + delta_mu_a)/(sqrt(2)*sigma_u))
    
    denom = 2
    denom += erf((delta_mu_c + delta_mu_a)/(sqrt(2)*sigma_p))
    denom -= erf((delta_mu_c - delta_mu_a)/(sqrt(2)*sigma_p))
    
    return num/denom


def di_theta_u(delta_mu_c, delta_mu_a, sigma_u, sigma_p):
    num = 2 
    
    denom = 2
    denom += erf((delta_mu_c + 2 * delta_mu_a)/(sqrt(2)*sigma_p))
    denom -= erf((delta_mu_c - 2 * delta_mu_a)/(sqrt(2)*sigma_p))
    
    return num/denom
    

def report(delta_mu_c, delta_mu_a, sigma_u, sigma_p, verbose=True):
    if verbose:
        print((2+erf((delta_mu_c - delta_mu_a)/(sqrt(2)*sigma_u)) - 
               erf((delta_mu_c + delta_mu_a)/(sqrt(2)*sigma_u)))/4)
        print(1)
        print('Positive Prediction Rate in Privileged Group (Optimal Classifier)')
        print((2+erf((delta_mu_c + delta_mu_a)/(sqrt(2)*sigma_p)) - 
               erf((delta_mu_c - delta_mu_a)/(sqrt(2)*sigma_p)))/4)
        print('Positive Prediction Rate in Unprivileged Group (Unprivileged Classifier)')
        print((2 + erf((delta_mu_c)/(sqrt(2)*sigma_p)) - 
               erf((delta_mu_c)/(sqrt(2)*sigma_p)))/4)
        print('Positive Prediction Rate in Privileged Group (Unprivileged Classifier)')
        print((2 + erf((delta_mu_c + 2 * delta_mu_a)/(sqrt(2)*sigma_p)) - 
               erf((delta_mu_c - 2 * delta_mu_a)/(sqrt(2)*sigma_p)))/4)
    print('DI(theta_u)')
    print(di_theta_u(delta_mu_c, delta_mu_a, sigma_u, sigma_p))
    print('DI(theta)')
    print(di_theta(delta_mu_c, delta_mu_a, sigma_u, sigma_p))

    
def plot_normal(mu, sigma, label=None):
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), label=label)


def plot_non_linear_boundary(mu1, mu2, sigma1, sigma2, p, d, label=None):
    x = np.linspace(-200, 200, 10000)
    y = np.log(p/(1-p)) - d*np.log(sigma1/sigma1) 
    y -= 1/(2*sigma1**2)*(x-mu1)**2 
    y += 1/(2*sigma2**2)*(x-mu2)**2
    plt.plot(x, y, label=label)
    plt.legend()


def get_c123(sigma_1, sigma_2, delta):
    sigma_1_theta_sqr = sigma_1 ** 2 + delta ** 2 / 16
    sigma_2_theta_sqr = sigma_2 ** 2 + delta ** 2 / 16

    denominator = sigma_1_theta_sqr ** 2 * sigma_2 ** 2
    denominator += sigma_2_theta_sqr ** 2 * sigma_1 ** 2
    denominator = 2 * np.sqrt(2) * delta * np.sqrt(denominator)
    c1 = delta ** 2 * sigma_2_theta_sqr / denominator
    c3 = delta ** 2 * sigma_1_theta_sqr / denominator
    c2 = 4 * sigma_1_theta_sqr * sigma_2_theta_sqr / denominator

    return c1, c2, c3


def get_selection_rate(sigma_1, delta, r, alpha, priv, is_tp):
    sigma_2 = r * sigma_1

    c_alpha = np.log(alpha / (1 - alpha))
    c1, c2, c3 = get_c123(sigma_1, sigma_2, delta)
    c = c1 if priv else c3
    c = c + c2 * c_alpha if is_tp else c - c2 * c_alpha

    return 0.5 + (1 if is_tp else -1) * 0.5 * erf(c)
