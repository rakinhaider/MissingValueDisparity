import pandas as pd
import numpy as np
from scipy import sparse
from .fair_dataset import FairDataset, default_mappings, _validate_alpha_beta
from imputations import impute


class MVDFairDataset(FairDataset):
    def __init__(self, n_samples, n_features=2, dist=None,
                 protected_attribute_names=['sex'],
                 privileged_group='Male',
                 label_name='label',
                 favorable_class=1,
                 classes=[0, 1],
                 sensitive_groups=['Female', 'Male'],
                 metadata=default_mappings,
                 alpha=0.5, beta=1, **kwargs):

        np.random.seed(kwargs.get('random_seed', 0))
        if dist is None:
            dist = FairDataset._get_default_dist()
        else:
            dist = self.get_detailed_dist(
                dist, n_features=n_features,
                protected_attribute_names=protected_attribute_names,
                privileged_group=privileged_group, alpha=alpha, beta=beta,
                favorable_class=favorable_class, classes=classes,
                sensitive_groups=sensitive_groups, **kwargs)
        self.complete_df = self.generate_synthetic(
            n_samples=n_samples, n_features=n_features, dist=dist,
            label_name=label_name, privileged_group=privileged_group,
            protected_attribute_names=protected_attribute_names,
            favorable_class=favorable_class, classes=classes,
            sensitive_groups=sensitive_groups, alpha=alpha, beta=beta, **kwargs)

        self.R = self.generate_missing_matrix(**kwargs)
        incomplete_df = self.get_incomplete_df(
            protected_attribute_names=protected_attribute_names)
        if kwargs.get('imputer', None):
            self.imputer = kwargs['imputer']
            if kwargs.get('keep_im_prot', False):
                non_feature_names = [] + [label_name]
            else:
                non_feature_names = protected_attribute_names + [label_name]
            feature_names = [i for i in incomplete_df.columns
                             if i not in non_feature_names]
            imputed_df = pd.DataFrame(
                self.imputer.transform(incomplete_df[feature_names]),
                columns=feature_names)
            imputed_df[non_feature_names] = incomplete_df[non_feature_names]
        elif 'method' in kwargs and kwargs['method'] == 'baseline':
            imputed_df = self.complete_df
            self.imputer = None
        else:
            imputed_df, self.imputer = impute(
                incomplete_df, kwargs.get('method', 'drop'),
                keep_im_prot=kwargs.get('keep_im_prot', False))

        super(MVDFairDataset, self).__init__(
            df=imputed_df, label_name=label_name,
            protected_attribute_names=protected_attribute_names,
            instance_weights_name=None, scores_names=[],
            unprivileged_protected_attributes=[[0]],
            privileged_protected_attributes=[[1]],
            favorable_label=favorable_class,
            unfavorable_label=1 - favorable_class,
            metadata=default_mappings, **kwargs
        )
        self.ignore_fields.add('complete_df')
        self.ignore_fields.add('R')

    def generate_synthetic(self, dist, **kwargs):
        # TODO: Handle alpha later
        # group_configs = self.get_group_configs(dist)
        parent = super(MVDFairDataset, self)
        return parent.generate_synthetic(dist, **kwargs)

    @property
    def imputed_df(self):
        df = pd.DataFrame(self.features, columns=self.feature_names)
        df[self.protected_attribute_names] = self.protected_attributes
        df[self.label_names] = self.labels
        df['index'] = self.instance_names
        df.set_index('index', drop=True, append=False, inplace=True)
        df.index.name = None
        return df

    def get_incomplete_df(self, label_names=['label'],
                          protected_attribute_names=['sex']):
        df = self.complete_df.copy(deep=True)
        n_features = df.shape[1] - 2
        s = df[protected_attribute_names]
        y = df[label_names]
        features = df.values[:, :n_features]
        rows = self.R.tocoo().row
        cols = self.R.tocoo().col
        features[rows, cols] = float('nan')
        df = pd.DataFrame(features, columns=df.columns[:-2])
        df[protected_attribute_names] = s
        df[label_names] = y
        return df

    def generate_missing_matrix(self, **kwargs):
        df = self.complete_df
        n_samples = df.shape[0]
        n_features = df.shape[1] - 2
        return np.zeros((n_samples, n_features))

    def get_detailed_dist(self, dist, **kwargs):
        raise NotImplementedError(
            "{:s}.get_formatted_dist is not implemented".format(
                __class__.__name__)
        )
