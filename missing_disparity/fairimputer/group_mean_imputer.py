import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Medium Blog https://towardsdatascience.com/coding-a-custom-imputer-in-scikit-learn-31bd68e541de

class GroupImputer(BaseEstimator, TransformerMixin):
    '''
    Class used for imputing missing values in a pd.DataFrame
    using either mean or median of a group.

    Parameters
    ----------
    group_cols : list
        List of columns used for calculating the aggregated value
    target : str
        The name of the column to impute
    metric : str
        The metric to be used for remplacement, can be one of ['mean', 'median']
    Returns
    -------
    X : array-like
        The array with imputed values in the target column
    '''

    def __init__(self, group_cols, target, metric='mean'):
        assert metric in ['mean',
                          'median'], \
            'Unrecognized value for metric, should be mean/median'
        assert type(
            group_cols) == list, 'group_cols should be a list of columns'
        assert type(target) == list, 'target should be a list of strings'

        self.group_cols = group_cols
        self.target = target
        self.metric = metric

    def fit(self, X, y=None):
        X = X.copy(deep=True)
        for c in self.group_cols:
            if c not in X.columns:
                self.group_cols.remove(c)
        assert pd.isnull(X[self.group_cols]).any(
            axis=None) == False, 'There are missing values in group_cols'

        X = X.dropna()
        groups = X.groupby(self.group_cols)
        impute_map = groups[self.target].agg(self.metric)
        impute_map.drop(columns=self.group_cols, inplace=True)
        impute_map.reset_index(drop=False, inplace=True)

        self.impute_map_ = impute_map

        return self

    def transform(self, X, y=None):
        # make sure that the imputer was fitted
        check_is_fitted(self, 'impute_map_')

        X = X.copy(deep=True)

        for index, row in self.impute_map_.iterrows():
            # print(X.head())
            # print(index, row)
            ind = (X[self.group_cols] == row[self.group_cols]).all(axis=1)
            # print(ind)
            X.loc[ind, self.target] = X.loc[ind, self.target].fillna(
                row[self.target])

        """print("impute_map")
        print(self.impute_map_)
        print("average")
        print(X.groupby(self.group_cols).agg(self.metric))
        """
        return X.values

