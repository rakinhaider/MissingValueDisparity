import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from fairimputer.group_mean_imputer import GroupImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def impute(df, method='drop', label_names=['label'],
           protected_attribute_names=['sex'], keep_im_prot=False, keep_y=False):

    feature_names_orig = df.columns
    s = df[protected_attribute_names]
    y = df[label_names]
    if keep_y:
        non_feature_names = []
    else:
        non_feature_names = label_names.copy()

    if not keep_im_prot:
        non_feature_names += protected_attribute_names

    feature_names = [i for i in df.columns
                     if i not in non_feature_names]
    features = df[feature_names]

    # print(feature_names)
    # print(features)
    if method == 'drop':
        # TODO: No longer balanced. Handle in later version.
        df.dropna(inplace=True)
        imputer = None
    else:
        if method == 'simple_imputer.mean':
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif method == 'iterative_imputer.mice':
            lr = LinearRegression()
            imputer = IterativeImputer(estimator=lr, max_iter=5,
                                       imputation_order='descending', random_state=0)
        elif method == 'iterative_imputer.missForest':
            rf = RandomForestRegressor()
            imputer = IterativeImputer(estimator=rf, max_iter=5,
                                       imputation_order='descending',
                                       random_state=0)
        elif method == 'knn_imputer':
            imputer = KNNImputer(n_neighbors=1, copy=True)
        elif method == 'group_imputer':
            imputer = GroupImputer(group_cols=protected_attribute_names,
                                   target=feature_names,
                                   metric='mean')
        else:
            raise ValueError("Imputation mechanism not supported.")
        df = pd.DataFrame(imputer.fit_transform(features),
                          columns=feature_names)

        if not keep_im_prot:
            df[protected_attribute_names] = s.values
        if not keep_y:
            df[label_names] = y.values

    df = df[feature_names_orig]
    return df, imputer
