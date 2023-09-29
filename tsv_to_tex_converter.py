import argparse
import pandas as pd

TYPICAL_REPLACEMENTS = {
    'baseline': 'Baseline',
    'drop': 'Drop',
    'simple_imputer.mean': 'Mean',
    'mean': 'Mean',
    'iterative_imputer.mice': 'MICE',
    'mice': 'MICE',
    'knn_imputer': 'k-NN',
    'knn': 'k-NN',
    'softimpute': 'Softimpute',
    'nb': 'NBC',
    'pr': 'PR'
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None)
    parser.add_argument('--init-tab', '-it', default=False,
                        action='store_true')
    args = parser.parse_args()
    path = args.path
    init_tab = args.init_tab
    df = pd.read_csv(path, sep='\t', header=None)
    df = df.fillna('')
    for index, row in df.iterrows():
        if init_tab:
            print('\t&\t', end='')

        if row[0] != '':
            print('\\midrule')
        row_vals = [f'{TYPICAL_REPLACEMENTS.get(c, c)}' for c in row]
        print('\t&\t'.join(row_vals) + '\\\\')
