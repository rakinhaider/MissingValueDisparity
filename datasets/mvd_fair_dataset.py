import pandas as pd
import numpy as np
from scipy import sparse
from .fair_dataset import FairDataset, default_mappings, _validate_alpha_beta
from imputations import impute


class MVDFairDataset(FairDataset):
    def __init__(self, n_samples, n_features,
                 protected_attribute_names=['sex'],
                 privileged_group='Male',
                 label_name='label',
                 favorable_class=1,
                 classes=[0, 1],
                 sensitive_groups=['Female', 'Male'],
                 group_shift=2, dist=None, metadata=default_mappings,
                 alpha=0.5, beta=1, **kwargs):

        print(kwargs)
        np.random.seed(kwargs.get('random_seed', 0))

        self.complete_df = self.generate_synthetic(
            n_samples=n_samples, n_features=n_features, label_name=label_name,
            protected_attribute_names=protected_attribute_names,
            privileged_group=privileged_group,
            favorable_class=favorable_class, classes=classes,
            sensitive_groups=sensitive_groups, dist=dist,
            group_shift=group_shift, alpha=alpha, beta=beta, **kwargs)

        self.R = self.generate_missing_matrix(**kwargs)
        incomplete_df = self.get_incomplete_df(
                    protected_attribute_names=protected_attribute_names)
        if kwargs.get('imputer', None):
            self.imputer = kwargs['imputer']
            non_feature_names = ([] if kwargs.get('keep_im_prot', False) else protected_attribute_names)+[label_name]
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

    def generate_synthetic(self, **kwargs):
        # TODO: Handle alpha later
        if kwargs['dist'] is None:
            return super(MVDFairDataset, self).generate_synthetic(**kwargs)
        group_configs = self.get_group_configs(**kwargs)
        parent = super(MVDFairDataset, self)
        return parent.generate_synthetic(group_configs=group_configs, **kwargs)

    def get_group_configs(self, **kwargs):
        mus_ = kwargs['dist']['mus']
        sigmas_ = kwargs['dist']['sigmas']
        group_configs = []
        for cls in kwargs['classes']:
            for sensitive_group in kwargs['sensitive_groups']:
                if sensitive_group != kwargs['privileged_group']:
                    shift, s = kwargs['group_shift'], 0
                else:
                    shift, s = 0, 1
                group_configs.append((mus_[cls] - shift, sigmas_, s, cls))
        return group_configs

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
        r = np.zeros((n_samples, n_features))
        r = sparse.csr_matrix(r)
        return r