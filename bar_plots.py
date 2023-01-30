import matplotlib.pyplot as plt
import pandas as pd

from utils import METHOD_SHORTS

if __name__ == "__main__":
    file = 'outputs/synthetic/ccd_none_nop_-3.tsv'
    f = open(file, 'r')
    all_splits = []
    for line in f:
        line = line[:-3]
        line = line.replace('$', '')
        splits = line.split('\t & \t')
        if len(splits) < 2:
            continue

        if splits[1] in ['method', 'baseline', 'drop', 'simple_imputer.mean', 'iterative_imputer.mice', 'knn_imputer']:
            all_splits.append(splits)

    all_splits = all_splits[:-1]
    df = pd.DataFrame(all_splits[1:], columns=all_splits[0])
    types = [float, str, float, float, float, float, float, float]
    df = df.astype({c: types[i] for i, c in enumerate(df.columns)})
    df['dis'] = (df['SR_p'] - df['SR_u']).abs()
    df['fpr_diff'] = (df['FPR_u'] - df['FPR_p']).abs()

    start_index = 0
    xticks = []
    for alpha in [0.25, 0.5, 0.75]:
        sub_df = df[df['alpha'] == alpha]
        methods = sub_df['method']

        plt.bar([i for i in range(start_index, start_index+ len(methods))],
                sub_df['dis'].values, label=alpha)
        xticks.extend(methods.values)
        xticks.append('')
        start_index += len(methods) + 1

    plt.xticks([i for i in range(start_index)],
               [METHOD_SHORTS.get(xt, xt) for xt in xticks], rotation=90)

    plt.tight_layout()
    plt.legend()
    plt.show()

    start_index = 0
    xticks = []
    for alpha in [0.25, 0.5, 0.75]:
        sub_df = df[df['alpha'] == alpha]
        methods = sub_df['method']

        plt.bar([i for i in range(start_index, start_index+ len(methods))],
                sub_df['fpr_diff'].values, label=alpha)
        xticks.extend(methods.values)
        xticks.append('')
        start_index += len(methods) + 1

    plt.xticks([i for i in range(start_index)],
               [METHOD_SHORTS.get(xt, xt) for xt in xticks], rotation=90)

    plt.tight_layout()
    plt.legend()
    plt.show()