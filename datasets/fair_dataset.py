import numpy as np
import pandas as pd

from aif360.metrics import (
    BinaryLabelDatasetMetric,
    ClassificationMetric
)
from aif360.datasets import StandardDataset, StructuredDataset
from itertools import product
from aif360.datasets.binary_label_dataset import BinaryLabelDataset
# TODO: The following mapping is only necessary for de_dummy operation.
# It is only used for generating all groups.
# Might consider removing it later.
default_mappings = {
    'label_maps': [{1.0: 'Yes', 0.0: 'No'}],
    'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'},
                                 {1.0: 'Caucasian', 0.0: 'Not Caucasian'}]
}


class FairDataset(BinaryLabelDataset):
    def __init__(self, **kwargs):
        """
            Args:
                See :obj:`StandardDataset` for a description of the arguments.
        """

        if kwargs['df'] is None:
            df = self.generate_synthetic(**kwargs)
        else:
            df = kwargs['df']

        super(FairDataset, self).__init__(
            df=df, label_names=[kwargs.get('label_name', str(df.columns[-1]))],
            protected_attribute_names=kwargs.get(
                'protected_attribute_names', str(df.columns[-2])),
            instance_weights_name=kwargs.get('instance_weights_name', None),
            metadata=kwargs.get('metadata', default_mappings))
        # TODO: Add assert of fairness

    def generate_synthetic(self, **kwargs):
        # TODO: Copy the previous code here.
        n_unprivileged = kwargs['n_samples']
        n_features = kwargs['n_features']
        alpha = kwargs['alpha']
        beta = kwargs['beta']
        _validate_alpha_beta(alpha, beta, n_unprivileged)
        if kwargs.get('group_configs', None):
            self.group_configs = kwargs['group_configs']
        elif kwargs.get('dist', None):
            self.group_configs = self.get_group_configs(**kwargs)
        else:
            self.group_configs = self._get_default_configs(**kwargs)
        protected_attribute_names = kwargs.get('protected_attribute_names',
                                               ['sex'])
        label_names = [kwargs.get('label_name', 'label')]
        columns = [str(i) for i in range(n_features)]
        columns += protected_attribute_names + label_names
        df = pd.DataFrame(columns=columns, dtype=float)

        for mus, sigmas, s, y in self.group_configs:
            n_samples = _get_n_samples(n_unprivileged, alpha, beta, s, y)
            columns = np.random.normal(mus, sigmas, (n_samples, n_features))
            sensitive_attribute = np.zeros((n_samples, 1)) + s
            class_column = np.zeros((n_samples, 1)) + y
            grp_data = np.concatenate(
                [columns, sensitive_attribute, class_column], axis=1)
            grp_df = pd.DataFrame(grp_data, columns=df.columns).astype(float)
            df = pd.concat([df, grp_df], axis=0, ignore_index=True)

        return df

    def get_group_configs(self, **kwargs):
        pass

    @staticmethod
    def _get_default_configs(**kwargs):
        mus_ = np.zeros(kwargs['n_features']) + 10
        sigmas_ = np.zeros((kwargs['n_features'])) + 3
        return [(mus_, sigmas_, 1, 1), (mus_ - 5, sigmas_, 1, 0),
                (mus_, sigmas_, 0, 1), (mus_ - 5, sigmas_, 0, 0),]

    def get_xy(self, keep_protected=False):
        x, _ = self.convert_to_dataframe()
        drop_fields = self.label_names.copy()
        if not keep_protected:
            drop_fields += self.protected_attribute_names

        x = x.drop(columns=drop_fields)

        y = self.labels.ravel()
        return x, y

    def _filter(self, columns, values):
        df, _ = self.convert_to_dataframe()
        for i, column in enumerate(columns):
            selection = None
            for val in values[i]:
                if selection:
                    selection = selection & (df[column] == val)
                else:
                    selection = df[column] == val

            df = df[selection]

        indices = [self.instance_names.index(i) for i in df.index]
        return self.subset(indices)

    def get_privileged_group(self):
        return self._filter(self.protected_attribute_names,
                            self.privileged_protected_attributes)

    def get_unprivileged_group(self):
        fd_priv = self._filter(self.protected_attribute_names,
                               self.privileged_protected_attributes)

        selected_rows = []
        for i in self.instance_names:
            if i not in fd_priv.instance_names:
                index = self.instance_names.index(i)
                selected_rows.append(index)

        return self.subset(selected_rows)

    @property
    def privileged_groups(self):
        return [{p: 1.0 for p in self.protected_attribute_names}]

    @property
    def unprivileged_groups(self):
        value_maps = self.metadata['protected_attribute_maps']
        value_maps = value_maps[:len(self.protected_attribute_names)]
        combination = product(*value_maps)
        groups = []
        for comb in combination:
            group = dict(zip(self.protected_attribute_names, comb))
            groups.append(group)
        for grp in self.privileged_groups:
            groups.remove(grp)
        return groups


def _validate_alpha_beta(alpha, beta, n_unprivileged):
    value_error = True
    if (int(beta * n_unprivileged) * alpha * 1.0).is_integer():
        if (n_unprivileged * alpha * 1.0).is_integer():
            value_error = False
    if value_error:
        raise ValueError("Number of positive or negative instances "
                         "must be integer. \nun_privileged * alpha "
                         "or int(beta * un_privileged) * alpha results "
                         "in floating values.")


def _get_n_samples(n_unprivileged, alpha, beta, s, y):
    n_samples = beta * n_unprivileged if s else n_unprivileged
    n_samples = alpha * n_samples if y else (1 - alpha) * n_samples
    return int(n_samples)


def _get_groups(attributes, value_maps):

    combinations = list(product(*value_maps))
    for comb in combinations:
        group = dict(zip(attributes, comb))
        for key in group:
            index = attributes.index(key)
            val = value_maps[index][group[key]]
            group[key] = val

        yield group


def _is_privileged(group, protected_attributes,
                   privileged_classes):

    for i in range(len(protected_attributes)):
        key = protected_attributes[i]
        if group[key] not in privileged_classes[i]:
            return False
    return True
