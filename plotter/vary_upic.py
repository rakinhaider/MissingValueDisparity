import pandas as pd
import matplotlib.pyplot as plt
from plotter.utils import set_size

if __name__ == '__main__':

    df = pd.read_csv('outputs/synthetic/vary_upic.tsv',
                     sep='\t', index_col=None)

    size = set_size(width=240, fraction=0.95, aspect_ratio=0.75)
    plt.figure(figsize=size, tight_layout=True)

    grouped = df.groupby('method')
    for tup, grp in grouped:
        print(tup)
        plt.plot(grp['upic'], grp['SR_p'], label=tup, linestyle='dashed')
        plt.plot(grp['upic'], grp['SR_u'], label=tup)

    handles, labels = plt.gca().get_legend_handles_labels()
    method_handles, method_labels = [], []
    legend_map = {'baseline': 'Baseline', 'drop': 'Drop',
                  'iterative_imputer.mice': 'MICE', 'knn_imputer': 'k-NN',
                  'simple_imputer.mean': 'Mean', 'softimpute': 'Softimpute'}
    for i in range(len(handles)):
        if i % 2 == 1:
            method_handles.append(handles[i])
            method_labels.append(legend_map[labels[i]])

    legend = plt.gca().legend(
        method_handles, method_labels, ncols=3, title='Methods',
        loc='upper center', fontsize=6, title_fontsize=6)

    handles, labels = plt.gca().get_legend_handles_labels()
    group_handles, group_labels = [handles[0], handles[1]], ['Priv.', 'Unpriv.']
    for h in group_handles:
        h.set_color('black')

    plt.legend(group_handles, group_labels, ncols=2,
               title='Groups', fontsize=6, title_fontsize=6, loc='center left',
               bbox_to_anchor=(0, 0.35))

    plt.gcf().add_artist(legend)

    plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlim(0.095, 0.605)
    plt.ylim(45, 62)
    plt.ylabel('Group-wise Selection Rates', fontsize=8)
    plt.xlabel('Rate of missing values\nin the unprivileged group', fontsize=8)
    plt.tight_layout()
    plt.savefig('outputs/synthetic/vary_upic_sr.pdf',
                format='pdf', bbox_inches='tight')
