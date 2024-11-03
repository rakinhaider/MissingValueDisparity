import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotter.utils import set_size

def plot_changes_by_y(what, y, loc='best', size=None):
    plt.figure(figsize=size, tight_layout=True)
    vals = []
    for gs, method in [(0, 'mean'), (-3, 'mean'), (-3, 'knn'), (-3, 'mice'), (3, 'mean')]:
        stats = pd.read_csv(f'outputs/synthetic/pred_changes/pred_changes_{gs}_{method}.tsv', sep='\t')
        stats.columns = ['s', 'y', 'rs'] + list(stats.columns[3:])
        stats.set_index(['s', 'y', 'rs'], inplace=True)
        for s in [0, 1]:
            # print(s, y)
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
    plt.ylim(vals.min() -  5, vals.max() + 25)
    plt.legend(loc=loc, fontsize=8)
    labels = ['(0, Mean)', '(-3, Mean)', '(-3, k-NN)', '(-3, MICE)', '(3, Mean)']
    plt.xticks(np.arange(5) + width/2, labels=labels, rotation=45, fontsize=9)
    plt.yticks(fontsize=9)
    if what in ['proba_less', 'proba_great']:
        # plt.ylabel(f'Perecntage of {"positive" if y == 1 else "negative"} individuals '
        #            f'with\nreceiving {"lower" if what == "proba_less" else "higher"} prediction probability.')
        ylabel = f'Perecntage of {"positive" if y == 1 else "negative"} individuals\n'
        if what == 'proba_less':
            ylabel += rf'with $\theta^{{\prime}}(x) < \theta(x)$'
        else:
            ylabel += rf'with $\theta^{{\prime}}(x) > \theta(x)$'
        plt.ylabel(ylabel, fontsize=9)

    else:
        plt.ylabel('Average change in rank.')
    plt.xlabel('(Group-Shift, Method)', fontsize=9)
    plt.tight_layout()
    os.makedirs('outputs/synthetic/pred_changes/figure/', exist_ok=True)
    plt.savefig(f'outputs/synthetic/pred_changes/figure/{what}_{y}.pdf', format='pdf')
    plt.show()
    plt.clf()


if __name__ == "__main__":
    size = set_size(240, 0.95, 1)
    plt.rcParams['text.usetex'] = True
    plot_changes_by_y('proba_great', 1, 'upper left', size)
    plot_changes_by_y('proba_less', 0, 'upper left', size)
