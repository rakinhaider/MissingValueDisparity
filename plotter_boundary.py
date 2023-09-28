import os
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from math import sqrt
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from plotter import set_rcparams, set_size

# Suppresing tensorflow warning

from utils import *


def plot_non_linear_boundary(mod, color='black', linestyles='solid'):
    xlim = (-15, 25)
    ylim = (-15, 25)
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = mod.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.gca().contour(xx, yy, Z, [0.5], colors=color, linestyles=linestyles)


def plot_boundary(df, baseline, mod, bmod):
    colors = [['tab:olive', 'tab:green'], ['tab:pink', 'tab:red']]
    grouped = df.groupby(by=['sex', 'label'])

    for (s, y), grp in grouped:
        s, y = int(s), int(y)
        c = colors[y][s]
        selected = grp.loc[np.random.choice(grp.index, 100)]
        plt.scatter(selected['0'], selected['1'], c=c, s=3)
        x_mean = grp['0'].mean()
        y_mean = grp['1'].mean()
        width = 2 * sqrt(5.991) * grp['0'].std()
        height = 2 * sqrt(5.991) * grp['1'].std()
        center = (x_mean, y_mean)
        # print(s, y, f"({'u' if s == 0 else 'p'}, "
        #             f"{'-' if y == 0 else '+' })")
        print(s, y, center, width, height)
        ellipse = Ellipse(xy=center, width=width, height=height,
                          color=c,
                          fill=False, linewidth=1,
                          label=f"({'u' if s == 0 else 'p'}, "
                                f"{'-' if y == 0 else '+' })")
        plt.gca().add_patch(ellipse)
    plot_non_linear_boundary(mod, linestyles='solid')
    plot_non_linear_boundary(bmod, linestyles='dotted')
    legends = plt.legend(loc='upper left', fontsize='xx-small')
    plt.xlabel(r'$x_1$', fontsize='x-small')
    plt.ylabel(r'$x_2$', fontsize='x-small')
    plt.xticks(fontsize='x-small')
    plt.yticks(fontsize='x-small')


if __name__ == "__main__":

    figsize = 241

    parser = get_parser()
    parser.add_argument('--distype', '-dt', default='ds_ccd',
                        choices=['ds_ccd', 'ccd', 'corr'],
                        help='Type of disparity')
    parser.add_argument('--priv-ic-prob', '-pic', default=0.1, type=float)
    parser.add_argument('--unpriv-ic-prob', '-upic', default=0.4, type=float)
    parser.add_argument('--group-shift', '-gs', default=0, type=int)
    parser.add_argument('--keep-im-prot', '-kip', default=False,
                        action='store_true',
                        help='Keep protected attribute in imputation')
    parser.add_argument('--keep-y', '-ky', default=False, action='store_true')
    parser.add_argument('--method', default='mean',
                        choices=['baseline', 'drop', 'mean',
                                 'mice', 'missForest', 'knn'])
    parser.add_argument('--test-method', '-tm', default='none',
                        choices=['none', 'train'])
    parser.add_argument('--header-only', default=False, action='store_true')
    args = parser.parse_args()

    protected = ["sex"]
    privileged_classes = [['Male']]

    LOG_FORMAT = '%(asctime)s - %(module)s - %(lineno)d - %(levelname)s \n %(message)s'
    logging.basicConfig(level=logging.WARN, format=LOG_FORMAT)

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
        dist = {'mus': {1: np.array([0 + class_shift,
                                     0 + class_shift + args.group_shift]),
                        0: np.array([0, 0 + args.group_shift])},
                'sigmas': [5, 5]}
    alpha = args.alpha
    method = METHOD_SHORT_TO_FULL[args.method]
    if method == "group_imputer":
        keep_prot = True
    else:
        keep_prot = args.keep_im_prot
    kwargs = {
        'protected_attribute_names': ['sex'], 'privileged_group': 'Male',
        'favorable_label': 1, 'classes': [0, 1],
        'sensitive_groups': ['Female', 'Male'], 'group_shift': args.group_shift,
        'beta': 1, 'dist': dist, 'keep_im_prot': keep_prot,
        'alpha': alpha, 'method': method, 'verbose': False,
        'priv_ic_prob': args.priv_ic_prob, 'unpriv_ic_prob': args.unpriv_ic_prob
    }
    estimator = get_estimator(args.estimator, args.reduce)
    keep_prot = args.reduce or (args.estimator == 'pr')
    n_samples = args.n_samples
    n_feature = args.n_feature
    test_method = None if args.test_method == 'none' else args.test_method

    train_fd, test_fd = get_synthetic_train_test_split(
        train_random_state=47, test_random_state=41, type=args.distype,
        n_samples=n_samples, n_features=n_feature,
        test_method=test_method, **kwargs)

    mod, _ = get_groupwise_performance(
        estimator, train_fd, test_fd, privileged=None)

    kwargs['method'] = 'baseline'
    baseline_train_fd, baseline_test_fd = get_synthetic_train_test_split(
        train_random_state=47, test_random_state=41, type=args.distype,
        n_samples=n_samples, n_features=n_feature,
        test_method=test_method, **kwargs)

    baseline_mod, _ = get_groupwise_performance(
        estimator, baseline_train_fd, baseline_test_fd, privileged=None)

    df, _ = test_fd.convert_to_dataframe()
    baseline_df, _ = baseline_test_fd.convert_to_dataframe()
    set_rcparams(fontsize=9)
    plt.figure(figsize=set_size(figsize, .95, 0.6),
               tight_layout=True)
    print(plt.gcf().get_size_inches())
    plot_boundary(df, baseline_df, mod, baseline_mod)
    plt.tight_layout()
    print(plt.gcf().get_size_inches())
    outdir = 'outputs/figures/boundary'
    os.makedirs(outdir, exist_ok=True)
    fname = f'{outdir}/{args.distype}_{args.delta}_{args.group_shift}' \
            f'{"" if args.test_method == "none" else "_train"}.pdf'
    plt.savefig(fname, format='pdf')
