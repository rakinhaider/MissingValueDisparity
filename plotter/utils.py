import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools


stat_to_str = {'acc': 'accuracy', 'sr': 'selection rates', 'fpr': 'false poisitive rates'}
tech_to_str = {'drop': 'Drop', 'mean': 'Mean', 'mice': 'MICE',
               'knn': 'k-NN', 'softimpute': 'Softimpute'}
out_dir = 'outputs/'


def plot_three_figures(df, db, grouping, size):
    for stat in ['acc', 'sr', 'fpr']:

        plt.figure(figsize=size)

        grouped = df.groupby(grouping)
        # bar_colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:olive', 'tab:brown']
        width = 0.85 / len(grouped)
        for i, (tup, grp) in enumerate(grouped):
            grp[f'sp_{stat}'] = grp[f'sp_{stat}'].apply(np.abs)
            baseline = grp[grp['method'] == 'baseline']
            grp = grp[grp['method'] != 'baseline']
            plt.bar(np.arange(len(grp)) + width * i , grp[f'sp_{stat}'] - baseline.iloc[0][f'sp_{stat}'], width=width,
                   label=tup)
        print(np.arange(len(grp)) + width * len(grouped) // 2, width * 1.0 * (len(grouped)//2))
        plt.xticks(np.arange(len(grp)) + width * (len(grouped) // 2), grp['method'], fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(title=grouping.title())
        plt.axhline(0, color='black', linestyle='dashed')
        plt.xlabel('Repair Technique', fontsize=14)
        plt.ylabel(f'Change of disparity in {stat_to_str[stat]}', fontsize=14)
        plt.tight_layout()
        target_dir = out_dir + f'standard/{db}/figures/{grouping}/'
        os.makedirs(target_dir, exist_ok=True)
        plt.savefig(f'{target_dir}/{stat}.pdf')
        plt.show()
        plt.clf()


def set_rcparams(**kwargs):
    fontsize = kwargs.get('fontsize', 10)
    params = {
        'text.usetex': True,
        'pdf.fonttype': 42,
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "font.size": fontsize,
        "axes.labelsize": 'medium',
        "axes.titlesize": 'medium',
        "xtick.labelsize": 'x-small',
        "ytick.labelsize": 'x-small',
        "mathtext.fontset": 'cm',
        "mathtext.default": 'bf',
        # "figure.figsize": set_size(width, fraction)
    }
    if kwargs.get('linewidth') is not None:
        params['lines.linewidth'] = kwargs.get('linewidth')
    if kwargs.get('titlepad') is not None:
        params["axes.titlepad"] = kwargs.get('titlepad')
    if kwargs.get('labelpad') is not None:
        params["axes.labelpad"] = kwargs.get('labelpad')
    if kwargs.get('markersize') is not None:
        params["lines.markersize"] = kwargs.get('markersize')
    matplotlib.rcParams.update(params)


# Source: https://tobiasraabe.github.io/blog/matplotlib-for-publications.html
def set_size(width, fraction=1, aspect_ratio='golden'):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    if aspect_ratio == 'golden':
        aspect_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * aspect_ratio

    return fig_width_in, fig_height_in