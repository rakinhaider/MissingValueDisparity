import logging
import argparse
import numpy as np

from datasets import DatasetFactory

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from aif360.metrics import ClassificationMetric
from aif360.datasets import GermanDataset, BankDataset, AdultDataset
from fairml import ExponentiatedGradientReduction, PrejudiceRemover
from aif360.algorithms.preprocessing.optim_preproc_helpers.\
    data_preproc_functions import \
    load_preproc_data_compas, load_preproc_data_german, load_preproc_data_adult


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
    if estimator == 'cat_nb':
        return CategoricalNB
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


def get_standard_dataset(dataset_name):
    if dataset_name == 'compas':
        dataset = load_preproc_data_compas(protected_attributes=['race'])
        dataset.privileged_groups = [{'race': 1}]
        dataset.unprivileged_groups = [{'race': 0}]
    elif dataset_name == 'german':
        """
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
        """
        dataset = load_preproc_data_german(protected_attributes=['age'])
        dataset.privileged_groups = [{'age': 1}]
        dataset.unprivileged_groups = [{'age': 0}]
    elif dataset_name == 'bank':
        dataset = BankDataset(
            # this dataset also contains protected
            protected_attribute_names=['age'],
            privileged_classes=[lambda x: x >= 25],  # age >=25 is considered privileged
        )
        dataset.privileged_groups = [{'age': 1}]
        dataset.unprivileged_groups = [{'age': 0}]
    elif dataset_name == 'adult':
        dataset = load_preproc_data_adult()
    else:
        raise ValueError('Dataset name must be one of '
                         'compas, german, bank')
    return dataset


def get_table_row(is_header=False, var_value=[], p_perf=None,
                  u_perf=None, m_perf=None, variable="alpha"):
    if isinstance(variable, tuple):
        variable = [str(v) for v in variable]
        row = [str(val) for val in var_value]
    else:
        variable = [str(variable)]
        row = [str(var_value)]
    cols = variable + ["AC_p", "AC_u", "SR_p", "SR_u", "FPR_p", "FPR_u"]
    if is_header:
        start = len(variable)
        cols[start:] = ["${:s}$".format(c) for c in cols[start:]]
        return "\t & \t".join(cols) + '\\\\'
    else:
        row += ["{:04.1f}".format(d) for d in [p_perf['AC_p'], u_perf['AC_u']]]
        row += ["{:04.1f}".format(m_perf[c]) for c in cols[len(variable)+2:]]
        return "\t & \t".join(row) + '\\\\'


def get_synthetic_train_test_split(type='ccd',
        train_random_state=47, test_random_state=23,
        test_method='train',
        **kwargs):
    factory = DatasetFactory()
    train_fd = factory.get_dataset(
        type=type, random_seed=train_random_state, **kwargs)
    test_kwargs = kwargs.copy()
    if test_method == 'train':
        test_fd = factory.get_dataset(type=type, random_seed=test_random_state,
                                      imputer=train_fd.imputer, **test_kwargs)
    else:
        if test_method is None:
            # print(test_kwargs['method'])
            test_kwargs['method'] = 'baseline'
        else:
            # NOTE: In case we want different train test imputations. Not
            # logical though.
            test_kwargs['method'] = test_method
        test_fd = factory.get_dataset(type=type, random_seed=test_random_state,
                                      **test_kwargs)

    return train_fd, test_fd


def filter(struct_data, columns, values):
    df, _ = struct_data.convert_to_dataframe()
    for i, column in enumerate(columns):
        selection = None
        for val in values[i]:
            if selection:
                selection = selection & (df[column] == val)
            else:
                selection = df[column] == val

        df = df[selection]

    indices = [struct_data.instance_names.index(i) for i in df.index]
    return struct_data.subset(indices)


def get_samples_by_group(struct_dataset, privileged):
    if privileged is None:
        return struct_dataset
    if privileged:
        values = struct_dataset.privileged_protected_attributes
    else:
        values = struct_dataset.unprivileged_protected_attributes
    return filter(
        struct_dataset, struct_dataset.protected_attribute_names, values)


def get_xy(data, keep_protected=False, keep_features='all'):
    x, _ = data.convert_to_dataframe()
    drop_fields = data.label_names.copy()
    if not keep_protected:
        drop_fields += data.protected_attribute_names

    x = x.drop(columns=drop_fields)

    if keep_features != 'all':
        x = x[keep_features]

    y = data.labels.ravel()
    return x, y


def get_groupwise_performance(train_fd, test_fd, estimator, privileged=None,
                              params=None, pos_rate=False, keep_features='all',
                              **kwargs):

    train_fd = get_samples_by_group(train_fd, privileged)

    if not params:
        params = get_model_params(estimator, train_fd)

    keep_prot = (estimator == ExponentiatedGradientReduction)
    keep_prot = keep_prot or (estimator == PrejudiceRemover)
    model = train_model(estimator, train_fd, params, keep_prot=keep_prot,
                        keep_features=keep_features, **kwargs)
    results = get_classifier_metrics(model, test_fd, verbose=False,
                                     sel_rate=pos_rate, keep_prot=keep_prot,
                                     keep_features=keep_features)

    return model, results


def get_model_params(model_type, train_fd):
    if model_type == SVC:
        params = {'probability': True}
    # elif model_type == CategoricalNB:
        # imputed_df, _ = get_xy(train_fd)
        # min_categories = []
        # for c in imputed_df.columns:
        #     sz = imputed_df[c].value_counts()
        #     # logging.info(c)
        #     # logging.info(sz)
        #     min_categories.append(sz.shape[0])
        # logging.info(list(zip(min_categories, imputed_df.columns)))
        # params = {'min_categories': min_categories}
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


def get_predictions(model, test_fd, keep_prot=False, keep_features='all',
                    **kwargs):
    test_fd_x, test_fd_y = get_xy(
        test_fd, keep_protected=keep_prot, keep_features=keep_features)
    logging.info(test_fd_x.shape)
    return model.predict(test_fd_x)


def train_model(model_type, data, params, keep_prot=False, keep_features='all',
                calibrate=None, calibrate_cv=10):
    x, y = get_xy(data, keep_protected=keep_prot, keep_features=keep_features)

    model = model_type(**params)
    if calibrate:
        model = CalibratedClassifierCV(model, method=calibrate, cv=calibrate_cv)
        logging.info(model)
    model = model.fit(x, y)
    return model


def get_classifier_metrics(clf, data, verbose=False, sel_rate=False,
                           keep_prot=False, keep_features='all'):
    unprivileged_groups = data.unprivileged_groups
    privileged_groups = data.privileged_groups

    data_pred = data.copy()
    data_x, data_y = get_xy(data, keep_protected=keep_prot,
                            keep_features=keep_features)
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


def KL_divergence(p, q, support=None):
    divs = []
    assert p.shape[0] == q.shape[0]
    for pi, qi in zip(p, q):
        qi = qi[:len(pi)]
        divs.append(np.sum(pi*np.log2(qi/pi)))
    logging.info(divs)
    return -sum(divs)