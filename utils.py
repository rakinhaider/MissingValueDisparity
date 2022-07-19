import argparse
import numpy as np
from copy import deepcopy
from datasets.dataset_factory import DatasetFactory
from aif360.algorithms.preprocessing.optim_preproc_helpers.\
    data_preproc_functions import \
    load_preproc_data_compas
from aif360.datasets import GermanDataset, BankDataset
from aif360.metrics import ClassificationMetric

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from fairml import ExponentiatedGradientReduction, PrejudiceRemover


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma-1', default=2, type=int)
    parser.add_argument('--sigma-2', default=5, type=int)
    parser.add_argument('--mu_p_plus', default=13, type=int)
    parser.add_argument('--mu_u_plus', default=10, type=int)
    parser.add_argument('--delta', default=10, type=int)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--beta', default=1, type=int)
    parser.add_argument('--n-samples', default=10000, type=int)
    parser.add_argument('--n-feature', default=2, type=int)
    parser.add_argument('--tr-rs', default=47, type=int,
                        help='Train Random Seed')
    parser.add_argument('--te-rs', default=41, type=int,
                        help='Test Random Seed')
    parser.add_argument('--estimator', default='nb',
                        choices=['nb', 'lr', 'svm', 'dt', 'pr'],
                        help='Type of estimator')
    parser.add_argument('--reduce', default=False, action='store_true')
    parser.add_argument('--print-header', default=False, action='store_true')
    return parser


def get_estimator(estimator, reduce):
    if reduce:
        return ExponentiatedGradientReduction

    if estimator == 'nb':
        return GaussianNB
    elif estimator == 'lr':
        return LogisticRegression
    elif estimator == 'svm':
        return SVC
    elif estimator == 'dt':
        return DecisionTreeClassifier
    elif estimator == 'nn':
        pass
    elif estimator == 'pr':
        return PrejudiceRemover


def get_dataset(dataset_name):
    if dataset_name == 'compas':
        dataset = load_preproc_data_compas(protected_attributes=['race'])
    elif dataset_name == 'german':
        dataset = GermanDataset(
            # this dataset also contains protected
            # attribute for "sex" which we do not
            # consider in this evaluation
            protected_attribute_names=['age'],
            # age >=25 is considered privileged
            privileged_classes=[lambda x: x >= 25],
            # ignore sex-related attributes
            features_to_drop=['personal_status', 'sex'],
        )
    elif dataset_name == 'bank':
        dataset = BankDataset(
            # this dataset also contains protected
            protected_attribute_names=['age'],
            privileged_classes=[lambda x: x >= 25],  # age >=25 is considered privileged
        )
        dataset.metadata['protected_attribute_maps'] = [{1.0: 'yes', 0.0: 'no'}]
        temp = dataset.favorable_label
        dataset.favorable_label = dataset.unfavorable_label
        dataset.unfavorable_label = temp
    else:
        raise ValueError('Dataset name must be one of '
                         'compas, german, bank')
    return dataset


def print_table_row(is_header=False, var_value=None, p_perf=None,
                    u_perf=None, m_perf=None, variable="alpha"):
    cols = [str(variable), "AC_p", "AC_u", "SR_p", "SR_u", "FPR_p", "FPR_u"]
    if is_header:
        print("\t & \t".join(cols))
    else:
        if isinstance(var_value, str) or isinstance(var_value, tuple):
            row = [str(var_value)]
        else:
            row = ['{:.2f}'.format(var_value)]
        row += ["{:04.1f}".format(d) for d in [p_perf['AC_p'], u_perf['AC_u']]]
        row += ["{:04.1f}".format(m_perf[c]) for c in cols[3:]]
        print("\t & \t".join(row))


def get_datasets(type='ccd', train_random_state=47, test_random_state=23,
                 **kwargs):
    factory = DatasetFactory()
    train_fd = factory.get_dataset(
        type=type, random_seed=train_random_state, **kwargs)
    test_fd = factory.get_dataset(type=type, random_seed=test_random_state,
                                  imputer=train_fd.imputer, **kwargs)
    return train_fd, test_fd


def get_groupwise_performance(train_fd, test_fd, estimator,
                              privileged=None,
                              params=None,
                              pos_rate=False,
                              privileged_group=None,
                              unprivileged_group=None):

    if privileged:
        train_fd = train_fd.get_privileged_group()

    elif privileged == False:
        train_fd = train_fd.get_unprivileged_group()

    if not params:
        params = get_model_params(estimator, train_fd)

    keep_prot = (estimator == ExponentiatedGradientReduction) or (estimator == PrejudiceRemover)
    model = train_model(estimator, train_fd, params, keep_prot=keep_prot)
    results = get_classifier_metrics(model, test_fd,
                                     verbose=False,
                                     sel_rate=pos_rate, keep_prot=keep_prot)

    return model, results


def get_model_params(model_type, train_fd):
    if model_type == SVC:
        params = {'probability': True}
    elif model_type == DecisionTreeClassifier:
        params = {'criterion': 'entropy',
                  'max_depth': 5,
                  'random_state': 47}
    elif model_type == LogisticRegression:
        params = {'class_weight': 'balanced',
                  'solver': 'liblinear'}
    elif model_type == GaussianNB:
        # params = {'priors':[0.1, 0.9]}
        params = {}
    elif model_type == ExponentiatedGradientReduction:
        params = {
            'prot_attr': train_fd.protected_attribute_names,
            'estimator': LogisticRegression(solver='liblinear'),
            'constraints': "DemographicParity",
            'drop_prot_attr': True
        }
    elif model_type == PrejudiceRemover:
        params = {
            'sensitive_attr': train_fd.protected_attribute_names[0],
            'class_attr': train_fd.label_names[0],
            'favorable_label': train_fd.favorable_label,
            'all_sensitive_attributes': train_fd.protected_attribute_names,
            'privileged_value': 1.0
        }
    else:
        params = {}
    return params


def get_model_performances(model, test_fd, pred_func,
                           keep_prot=False, **kwargs):
    data = test_fd.copy()
    data_pred = test_fd.copy()
    data_pred.labels = pred_func(model, test_fd, keep_prot=keep_prot, **kwargs)

    metrics = ClassificationMetric(
        data, data_pred, privileged_groups=test_fd.privileged_groups,
        unprivileged_groups=test_fd.unprivileged_groups)

    perf = {}
    perf['SR'] = metrics.selection_rate()

    perf['AC_p'] = metrics.accuracy(privileged=True)
    perf['SR_p'] = metrics.selection_rate(privileged=True)
    perf['TPR_p'] = metrics.true_positive_rate(privileged=True)
    perf['FPR_p'] = metrics.false_positive_rate(privileged=True)

    perf['AC_u'] = metrics.accuracy(privileged=False)
    perf['SR_u'] = metrics.selection_rate(privileged=False)
    perf['TPR_u'] = metrics.true_positive_rate(privileged=False)
    perf['FPR_u'] = metrics.false_positive_rate(privileged=False)
    for k in perf:
        perf[k] = perf[k] * 100
    return perf


def get_predictions(model, test_fd, keep_prot=False):
    test_fd_x, test_fd_y = test_fd.get_xy(keep_protected=keep_prot)
    return model.predict(test_fd_x)


def train_model(model_type, data, params, keep_prot=False):
    x, y = data.get_xy(keep_protected=keep_prot)

    model = model_type(**params)
    model = model.fit(x, y)

    return model


def get_classifier_metrics(clf, data,
                           verbose=False,
                           sel_rate=False, keep_prot=False):
    unprivileged_groups = data.unprivileged_groups
    privileged_groups = data.privileged_groups

    data_pred = data.copy()
    data_x, data_y = data.get_xy(keep_protected=keep_prot)
    data_pred.labels = clf.predict(data_x)
    metrics = ClassificationMetric(data,
                                   data_pred,
                                   privileged_groups=privileged_groups,
                                   unprivileged_groups=unprivileged_groups)

    if verbose:
        print('Mean Difference:', abs(0 - metrics.mean_difference()))
        print('Disparate Impact:', abs(1 - metrics.disparate_impact()))
        # print('Confusion Matrix:', metrics.binary_confusion_matrix())
        print('Accuracy:', metrics.accuracy())

    m = [metrics.mean_difference(),
         metrics.disparate_impact(),
         abs(1 - metrics.disparate_impact()),
         metrics.accuracy(),
         metrics.accuracy(privileged=True),
         metrics.accuracy(privileged=False)]

    if sel_rate:
        for pg in [True, False]:
            for sr in [True, False]:
                m.append(get_positive_rate(metrics, pg, sr))

    return m


def get_positive_rate(cmetrics, privileged=None, positive=True):
    if positive:
        return cmetrics.true_positive_rate(privileged)
    else:
        return cmetrics.false_positive_rate(privileged)
