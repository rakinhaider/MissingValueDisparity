import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotter.utils import set_size


def plot_changes_by_y_real(what, y, loc='best', size=None):
    plt.figure(figsize=size, tight_layout=True)
    vals = []
    dbs = ['compas', 'folkincome', 'german', 'pima', 'heart']
    for db in dbs:
        stats = pd.read_csv(f'outputs/standard/{db}/pred_changes/pred_changes_{db}_1.tsv', sep='\t')
        stats.columns = ['s', 'y', 'rs'] + list(stats.columns[3:])
        if db == 'german':
            stats['y'] = stats['y'] - 1
        stats.set_index(['s', 'y', 'rs'], inplace=True)
        for s in [0, 1]:
            print(db, s, y)
            group = stats.loc[s, y, :]
            vals.append(group[what].mean())

    # print(vals)

    width = 0.3  # the width of the bars
    multiplier = 0
    rects = plt.bar(np.arange(5), vals[1:10:2], width, label='Privileged')
    plt.bar_label(rects, padding=3, rotation=90, fontsize=8, fmt='%0.2f')

    offset = width * 1
    rects = plt.bar(np.arange(5) + offset, vals[0:10:2], width, label='Unprivileged')
    plt.bar_label(rects, padding=3, rotation=90, fontsize=8, fmt='%0.2f')

    vals = np.array(vals)
    plt.ylim(vals.min() -  5, vals.max() + 15)
    plt.legend(loc=loc, fontsize=8)
    db_name_map = {'compas': 'COMPAS', 'folkincome': 'FolkIncome',
                   'german': 'German', 'pima': 'PIMA', 'heart': 'Heart'}
    plt.xticks(np.arange(5) + width/2,
               [db_name_map[d] for d in dbs], fontsize=10, rotation=45)
    plt.yticks(fontsize=10)
    if what in ['proba_less', 'proba_great']:
        # plt.ylabel(f'Perecntage of {"positive" if y == 1 else "negative"} individuals '
        #            f'with\nreceiving {"lower" if what == "proba_less" else "higher"} prediction probability.')
        ylabel = f'Perecntage of {"positive" if y == 1 else "negative"} individuals\n'
        if what == 'proba_less':
            ylabel += rf'with $\theta^{{\prime}}(x) < \theta(x)$'
        else:
            ylabel += rf'with $\theta^{{\prime}}(x) > \theta(x)$'
        plt.ylabel(ylabel, fontsize=10)
    plt.xlabel('(Group-Shift, Method)', fontsize=10)
    plt.tight_layout()
    os.makedirs('outputs/standard/figures/pred_changes', exist_ok=True)
    plt.savefig(f'outputs/standard/figures/pred_changes/{what}_{y}.pdf', format='pdf')
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    size = set_size(240, 0.95, 1)
    plt.rcParams['text.usetex'] = True
    plot_changes_by_y_real('proba_great', 1, 'upper left', size=size)
    plot_changes_by_y_real('proba_less', 0.0, 'upper right', size=size)