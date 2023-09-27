import numpy as np
import pandas as pd


def missing_by_top_corr_column(df, uic, pic, pa_names, label_names, rng):
    col_name, col_index = get_most_corr_column(df, label_names, pa_names)
    return get_missing_matrix_by_column(df, uic, pic, pa_names, col_index, rng)


def missing_by_column(df, uic, pic, pa_names, col_name, rng):
    col_index = df.columns.get_loc(col_name)
    return get_missing_matrix_by_column(df, uic, pic, pa_names, col_index, rng)


def missing_single_col_by_sample(df, uic, pic, pa_names, label_names, rng):
    return get_missing_matrix_by_samples(df, uic, pic, label_names, pa_names,
                                         rand_single_col_selector, rng)


def missing_all_cols_by_sample(df, uic, pic, pa_names, label_names, rng):
    return get_missing_matrix_by_samples(
        df, uic, pic, label_names, pa_names, rand_many_col_selector, rng)


def get_most_corr_column(df, label_names, pa_names):
    corr = df.corr()[label_names[0]]
    corr = np.abs(corr)
    corr = corr.dropna()
    corr = corr.drop(index=label_names[0])
    corr = corr.drop(index=pa_names)
    missing_column_name = corr.idxmax()
    missing_column_index = df.columns.get_loc(missing_column_name)
    return missing_column_name, missing_column_index


def get_missing_matrix_by_column(df, uic, pic, pa_names, col_index, rng):
    n_samples = df.shape[0]
    n_features = df.shape[1] - 1
    missing_matrix = np.zeros((n_samples, n_features))
    for r, p in zip([0, 1], [uic, pic]):
        selector = df[pa_names[0]] == r
        indices = [i for i, s in enumerate(selector.values) if s]
        rng.shuffle(indices)
        n_missing = int(np.ceil(p * len(indices)))
        choices = rng.choice(indices, size=n_missing,
                                   replace=False,)
        missing_matrix[choices, col_index] += 1

    return missing_matrix


def get_missing_matrix_by_samples(df, uic, pic, label_names, pa_names,
                                  column_selector, rng, **kwargs):
    n_samples = df.shape[0]
    n_features = df.shape[1] - 1
    missing_matrix = np.zeros((n_samples, n_features))

    for i, (index, row) in enumerate(df.iterrows()):
        if have_missing(row, uic, pic, pa_names, rng):
            selected_columns = column_selector(
                row, uic, pic, pa_names, label_names, rng, **kwargs)
            missing_matrix[i, selected_columns] = 1
    return missing_matrix


def have_missing(row, uic, pic, pa_names, rng):
    toss = rng.rand()
    if row[pa_names[0]] == 1:
        return toss >= (1 - pic)
    else:
        return toss >= (1 - uic)


def rand_single_col_selector(row, uic, pic, pa_names, label_names, rng):
    """
    :return: Returns index of a single column where the missing value will be
    """
    columns = list(row.index)
    indices = [i for i, c in enumerate(columns) if c not in pa_names+label_names]
    return rng.choice(indices, size=1, replace=False)


def rand_many_col_selector(row, uic, pic, pa_names, label_names, rng, cutoff=0.5):
    """
        :return: Returns random list of indices of column
            where the missing value will be
    """
    columns = list(row.index)
    indices = [i for i, c in enumerate(columns) if
               c not in pa_names + label_names]
    toss = rng.rand(len(indices))
    return [i for i, t in zip(indices, toss >= cutoff) if t]




