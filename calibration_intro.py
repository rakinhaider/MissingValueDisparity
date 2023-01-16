import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.calibration import calibration_curve, CalibrationDisplay, CalibratedClassifierCV
from utils import get_xy
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from aif360.datasets.bank_dataset import BankDataset
from aif360.datasets.german_dataset import GermanDataset
from aif360.datasets.adult_dataset import AdultDataset

if __name__ == "__main__":
    # data = load_preproc_data_compas()
    # data = BankDataset()
    # data = GermanDataset()
    data = AdultDataset()
    dataset_name = 'adult'
    train, test = data.split([0.8], shuffle=True, seed=47)

    x, y = get_xy(train, keep_protected=False)
    test_x, test_y = get_xy(test, keep_protected=False)

    gnb = GaussianNB()
    gnb.fit(x, y)

    clf_list = [
        # [CategoricalNB(), 'cat_nb'],
        [GaussianNB(), 'gnb'],
        [LogisticRegression(max_iter=10000), 'logistic'],
        [CalibratedClassifierCV(gnb, method='sigmoid', cv=10), 'cali_sigmoid'],
        [CalibratedClassifierCV(gnb, method='isotonic', cv=10), 'cali_isotonic'],
    ]

    gs = GridSpec(4, 2)
    calibration_displays = {}
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gcf().add_subplot(gs[:2, :2])
    colors = plt.cm.get_cmap("Dark2")
    for i, [clf, name] in enumerate(clf_list):
        clf.fit(x, y)
        display = CalibrationDisplay.from_estimator(
            clf,
            test_x,
            test_y,
            n_bins=20,
            name=name,
            ax=ax,
            color=colors(i)
        )
        calibration_displays[name] = display

    # y_pred = clf_list[2][0].predict_proba(test_x)
    # print(calibration_curve(test_y, y_pred[:, 1], n_bins=10))

    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
    for i, (_, name) in enumerate(clf_list):
        row, col = grid_positions[i]
        ax = plt.gcf().add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=20,
            label=name,
            color=colors(i)
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.savefig('cali_curve_' + dataset_name + '.pdf', format='pdf')
    plt.show()