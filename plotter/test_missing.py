import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotter.utils import stat_to_str, set_size, out_dir, tech_to_str


if __name__ == "__main__":

    df = pd.read_csv('outputs/synthetic/tables/corr_-3_tm.tsv', sep='\t', header=None)
    df.columns = ['method', 'sp_acc', 'sp_acc_var', 'sp_sr', 'sp_sr_var', 'sp_fpr', 'sp_fpr_var']

    size = set_size(240, 0.95, 0.85)

    for stat in ['acc', 'sr', 'fpr']:
        plt.figure(figsize=size)

        grouped = [('', df)]
        bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:olive', 'tab:brown']
        width = 0.85 / len(grouped)
        less_than_zero = False
        for i, (tup, grp) in enumerate(grouped):
            grp[f'sp_{stat}'] = grp[f'sp_{stat}'].apply(np.abs)
            baseline = grp[grp['method'] == 'baseline']
            grp = grp[grp['method'] != 'baseline']
            print(tup, stat)
            print((grp[f'sp_{stat}'] - baseline.iloc[0][f'sp_{stat}']).min())
            if (grp[f'sp_{stat}'] - baseline.iloc[0][f'sp_{stat}']).min() < 0:
                less_than_zero = True
            print(less_than_zero)
            plt.bar(np.arange(len(grp)) + width * i, grp[f'sp_{stat}'] - baseline.iloc[0][f'sp_{stat}'],
                    color=bar_colors)
        print(np.arange(len(grp)) + width * len(grouped) // 2, width * 1.0 * (len(grouped) // 2))
        plt.xticks(np.arange(len(grp)) + width * (len(grouped) // 2),
                   [tech_to_str[m] for m in grp['method']], fontsize=9, rotation=20)
        plt.yticks(fontsize=9)
        if less_than_zero:
            plt.axhline(0, color='black', linestyle='dashed')
        plt.xlabel('Repair Technique', fontsize=10)
        plt.ylabel(f'Change of disparity in\n{stat_to_str[stat]}', fontsize=10)
        plt.tight_layout()
        target_dir = out_dir + f'synthetic/figures/test_missing/'
        os.makedirs(target_dir, exist_ok=True)
        plt.savefig(f'{target_dir}/{stat}.pdf')
        plt.show()
        plt.clf()