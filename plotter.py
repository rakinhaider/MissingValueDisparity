import logging
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from utils import get_parser, get_estimator, get_synthetic_train_test_split, METHOD_SHORTS


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


def plot_normal(mu, sigma, ax, label=None, privileged=None, cls=0):
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    if cls == 1:
        color = '#1f77b4'
        linestyle = '--'
    else:
        color = '#ff7f0e'
        linestyle = (0, (1, 1))
    label = '+' if cls else '-'
    ax.plot(x, stats.norm.pdf(x, mu, sigma), linestyle=linestyle,
            color=color, label=label)


def plot_feat_dist(dist, **kwargs):
    width = kwargs.get('width', 462)
    fraction = kwargs.get('fraction', 1)
    aspect_ratio = kwargs.get('aspect_ratio', 'golden')
    f, ax = plt.subplots(2, 2,
        figsize=set_size(width, fraction, aspect_ratio=aspect_ratio)
    )

    mus = dist['mus']
    sigmas = dist['sigmas']
    range_min = np.min(mus-3*sigmas, axis=1)
    range_max = np.max(mus+3*sigmas, axis=1)

    for i, row in enumerate(ax):
        for j, a in enumerate(row):
            for l in [0, 1]:
                plot_normal(mus[i][l][j], sigmas[i][l][j], a)
                # left, right = range_min[i][]
                # a.set_xticks([int(i) for i in np.linspace(left, right, 5)])
                # a.set_yticks([0, 0.1, 0.2])
                # a.set_xlim(left, right)
                # a.set_ylim(-0.01, 0.22)
                print(mus[i][l][j], sigmas[i][l][j], i, j)

    ax[0][0].set_xlabel(r'$x_1$')
    ax[0][0].set_ylabel(r'$\mathcal{N}(\mu_1, \sigma_1)$')
    ax[0][1].set_xlabel(r'$x_1$')
    ax[0][1].set_ylabel(r'$\mathcal{N}(\mu_1, \sigma_1)$')

    ax[1][0].set_xlabel(r'$x_2$')
    ax[1][0].set_ylabel(r'$\mathcal{N}(\mu_2, \sigma_2)$')
    ax[1][1].set_xlabel(r'$x_2$')
    ax[1][1].set_ylabel(r'$\mathcal{N}(\mu_2, \sigma_2)$')

    ax[0][0].set_title('Privileged')
    ax[0][1].set_title('Unprivileged')

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.25)


def get_range(dist_table, i, j):
    mus = [l[j] for l in dist_table[:, :, 0].flatten()]
    sigmas = [l[j] for l in dist_table[:, :, 1].flatten()]
    vals = [mu - 3 * sigmas[i] for i, mu in enumerate(mus)]
    vals += [mu + 3 * sigmas[i] for i, mu in enumerate(mus)]
    left = int(round(min(vals), 0))
    right = int(round(max(vals), 0))
    return left, right


def plot_dist_table(dist_table, **kwargs):
    width = kwargs['width']
    fraction = kwargs['fraction']
    aspect_ratio = kwargs['aspect_ratio']
    f, ax = plt.subplots(2, 2,
            figsize=set_size(width, fraction, aspect_ratio=aspect_ratio))
    print(dist_table)
    # i controls sensitive attribute
    for i in range(2):
        # j controls class
        for j in range(2):
            mus = dist_table[i][j][0]
            sigmas = dist_table[i][j][1]
            plot_normal(float(mus[0]), float(sigmas[0]), ax[i][0],
                        privileged=i, cls=j)
            plot_normal(float(mus[1]), float(sigmas[1]), ax[i][1],
                        privileged=i, cls=j)

    for i in range(2):
        for j in range(2):
            left, right = get_range(dist_table, j, i)
            ax[j][i].set_xlim(left, right)
            ax[j][i].set_xticks(
                [i for i in range(left, right + 1) if i % 10 == 0])
            ax[j][i].set_ylim(-0.01, 0.125)
            # ax[j][i].legend(fontsize=5)

    ax[0][0].set_title(r"$x_1$")
    ax[1][0].set_title(r"$x_1$")
    ax[0][1].set_title(r"$x_2$")
    ax[1][1].set_title(r"$x_2$")

    ax[1][0].set_ylabel('Privileged')
    ax[0][0].set_ylabel('Unprivileged')

    # plt.subplots_adjust(wspace=0.25)
    plt.tight_layout()


def plot_group_config(group_config, **kwargs):
    dist_table = np.reshape(group_config, (2, 2, 4))
    plot_dist_table(dist_table, **kwargs)


def get_df_group_config(df, precision=3):
    df_gc = []

    for s in [0, 1]:
        for l in [0, 1]:
            desc = df[((df['sex'] == s) & (df['label'] == l))].describe()
            mus = desc.loc['mean']
            sigmas = desc.loc['std']
            df_gc.append((
                [round(mus[0], precision), round(mus[1], precision)],
                [round(sigmas[0], precision), round(sigmas[1], precision)], s,
                l)
            )
    print(*df_gc, sep='\n')
    return df_gc


if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument('--distype', '-dt', default='ds_ccd',
                        choices=['ds_ccd', 'ccd', 'corr'],
                        help='Type of disparity')
    parser.add_argument('--group-shift', '-gs', default=0, type=int)
    parser.add_argument('--priv-ic-prob', '-pic', default=0.1, type=float)
    parser.add_argument('--unpriv-ic-prob', '-upic', default=0.4, type=float)
    parser.add_argument('--test-method', '-tm', default='train',
                        choices=['none', 'train'])
    parser.add_argument('--width', default=241)
    parser.add_argument('--fontsize', default=9)
    parser.add_argument('--log-level', '-ll', default='WARN')
    parser.add_argument('--what', default='featdist',
                        choices=['featdist', 'byupic', 'barplot'])
    args = parser.parse_args()

    protected = ["sex"]
    privileged_classes = [['Male']]

    LOG_FORMAT = '%(asctime)s - %(module)s - %(lineno)d - %(levelname)s \n %(message)s'
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT)

    if args.what == 'featdist':
        # Class shift is 10
        class_shift = args.delta
        if args.distype == 'corr':
            group_shift = args.group_shift
            dist = {
                'mus': {'x1': {
                    0: [0, 0 + group_shift],
                    1: [0 + class_shift, 0 + group_shift + class_shift]},
                    'z': [0, 2]},
                'sigmas': {'x1': {0: [5, 5], 1: [5, 5]}, 'z': [1, 1]},
            }
        else:
            dist = {'mus': {
                1: np.array(
                    [0 + class_shift, 0 + class_shift + args.group_shift]),
                0: np.array([0, 0 + args.group_shift])},
                'sigmas': [5, 5]}
        alpha = args.alpha
        method = 'baseline' if args.test_method == 'none' else 'simple_imputer.mean'
        kwargs = {
            'protected_attribute_names': ['sex'], 'privileged_group': 'Male',
            'favorable_label': 1, 'classes': [0, 1],
            'sensitive_groups': ['Female', 'Male'],
            'group_shift': args.group_shift,
            'beta': 1, 'dist': dist, 'alpha': alpha, 'method': method,
            'verbose': False, 'priv_ic_prob': args.priv_ic_prob,
            'unpriv_ic_prob': args.unpriv_ic_prob
        }
        estimator = get_estimator(args.estimator, args.reduce)
        keep_prot = args.reduce or (args.estimator == 'pr')
        n_samples = args.n_samples
        n_feature = args.n_feature
        test_method = None if args.test_method == 'none' else args.test_method

        logging.info(kwargs)
        # TODO: ############ Results not matching with notebooks ##############
        train_fd, test_fd = get_synthetic_train_test_split(
            train_random_state=47, test_random_state=41, type=args.distype,
            n_samples=n_samples, n_features=n_feature,
            test_method=test_method, **kwargs)

        # textwidth = 505
        linewidth = args.width
        out_dir = '../outputs/figures/'

        set_rcparams(fontsize=args.fontsize)
        if not test_method:
            group_configs = test_fd.group_configs
        else:
            group_configs = get_df_group_config(test_fd.imputed_df)

        print(group_configs)
        plot_group_config(group_configs, width=args.width,
                          fraction=0.95, aspect_ratio=.65)
        fname = '{:s}_{:d}_{:d}_{:s}.pdf'.format(args.distype, args.delta,
                                                 args.group_shift,
                                                 args.test_method)

        print(plt.gcf().get_size_inches())
        plt.savefig(out_dir + fname, format='pdf')
    elif args.what == 'byupic':
        fname = 'outputs/synthetic/vary_upic.tsv'
        splits = []
        for line in open(fname, 'r'):
            line = line.replace('$', '')
            line = line.replace('\\\\\n', '')
            split = line.split('\t & \t')
            if len(split) > 2:
                splits.append(split)

        df = pd.DataFrame(splits[1:], columns=splits[0])
        types = [str, float, float, float, float, float, float]
        df = df.astype({c: types[i] for i, c in enumerate(df.columns)})
        print(df)

        pad = 1.2
        bbox_to_anchor_top = 1.12

        set_rcparams(fontsize=9, linewidth=1)
        fig = plt.figure(figsize=set_size(200, .95, 0.6))

        styles = ['--', '-']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        for i, col in enumerate(['FPR_p', 'FPR_u']):
            plt.hlines(y=df.loc[0][col], xmin=0, xmax=1, linestyles=styles[i],
                       label='baseline', color='black')
            for j, method in enumerate(['drop', 'mean', 'mice', 'knn']):
                sub_df = df[df['method'] == method]
                print(sub_df[col])
                plt.plot([i / 10 for i in range(1, 7)], sub_df[col],
                         styles[i], color=colors[j],
                         label='{}'.format(method))
                # axs[i].set_xlabel(r'${}$'.format(col))
                plt.xlabel(r'$\lambda_u$', fontsize=9)
                plt.ylabel(r'$FPR_s$', fontsize=9)

            plt.xlim(0.08, 0.62)
            plt.xticks([i/10 for i in range(1, 7)])

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.figlegend(handles[5:], labels[5:],
                      bbox_to_anchor=(0.58, bbox_to_anchor_top),
                      loc='upper center', ncol=3, fontsize='xx-small')
        plt.legend([handles[0], handles[5]], [r'$FPR_p$', r'$FPR_u$'],
                   fontsize='xx-small', loc='center right')
        plt.tight_layout()
        plt.savefig('outputs/figures/vary_upic_fpr.pdf', format='pdf',
                    bbox_inches='tight')


        # TODO: update SR plot accordingly
        set_rcparams(fontsize=9)
        fig, axs = plt.subplots(
            1, 2, figsize=set_size(242, .95, 0.45))

        for i, col in enumerate(['SR_p', 'SR_u']):
            axs[i].hlines(y=df.loc[0][col], xmin=0, xmax=1, linestyles='dashed',
                          label='baseline', color='black')
            for method in ['drop', 'mean',
                           'mice', 'knn']:
                sub_df = df[df['method'] == method]
                print(sub_df[col])
                axs[i].plot([i / 10 for i in range(1, 7)], sub_df[col],
                            # '-^', markersize=3,
                            label='{}'.format(method))
                axs[i].set_xlabel(col)

            axs[i].set_xlim(0.8, 6.2)

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.figlegend(handles[:5], labels[:5],
                      bbox_to_anchor=(0.5, bbox_to_anchor_top),
                      loc='upper center', ncol=5, fontsize='xx-small')
        plt.tight_layout()
        plt.savefig('outputs/figures/vary_upic_sr.pdf', format='pdf',
                    bbox_inches='tight')
