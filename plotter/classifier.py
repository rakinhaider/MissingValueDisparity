import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotter.utils import stat_to_str, set_size, out_dir, tech_to_str


if __name__ == "__main__":
    db = 'folkincome'
    df = pd.read_csv(f'outputs/standard/{db}/tables/sota_cls_perf.tsv', sep='\t', header=None)
    df.columns = ['classifiers'] + ['method', 'sp_acc', 'sp_acc_var', 'sp_sr', 'sp_sr_var', 'sp_fpr', 'sp_fpr_var']
    classifiers = [[i] * 6 for i in ['NB', 'SVM', 'LR', 'DT', 'NN', 'PR', 'RBC']]
    df['classifiers'] = sum(classifiers, [])
    df = df[df['classifiers'] != 'LR']
    df = df[df['classifiers'] != 'DT']

    size = set_size(240, 0.95, 0.85)
    grouping = 'classifiers'
    for stat in ['acc', 'sr', 'fpr']:

        plt.figure(figsize=size)

        grouped = df.groupby(grouping)
        # bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:olive', 'tab:brown']
        width = 0.85 / len(grouped)
        maxval = 0
        for i, (tup, grp) in enumerate(grouped):
            grp[f'sp_{stat}'] = grp[f'sp_{stat}'].apply(np.abs)
            baseline = grp[grp['method'] == 'baseline']
            grp = grp[grp['method'] != 'baseline']
            plt.bar(np.arange(len(grp)) + width * i , grp[f'sp_{stat}'] - baseline.iloc[0][f'sp_{stat}'], width=width,
                   label=tup)
            maxval = max(maxval, (grp[f'sp_{stat}'] - baseline.iloc[0][f'sp_{stat}']).max())
        print(np.arange(len(grp)) + width * len(grouped) // 2, width * 1.0 * (len(grouped)//2))
        plt.ylim(top=maxval+2)
        plt.xticks(np.arange(len(grp)) + width * (len(grouped) // 2),
                   [tech_to_str[m] for m in grp['method']], fontsize=9,
                   rotation=20)
        plt.yticks(fontsize=9)
        plt.legend(title=grouping.title(), title_fontsize=8,
                   fontsize=8, ncols=3)
        plt.axhline(0, color='black', linestyle='dashed')
        plt.xlabel('Repair Technique', fontsize=10)
        plt.ylabel(f'Change of disparity in\n {stat_to_str[stat]}',
                   fontsize=10)
        plt.tight_layout(pad=0.2)
        target_dir = out_dir + f'standard/{db}/figures/{grouping}/'
        os.makedirs(target_dir, exist_ok=True)
        plt.savefig(f'{target_dir}/{stat}.pdf')
        plt.show()
        plt.clf()
