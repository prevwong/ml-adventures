from __future__ import print_function

from time import time

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets.base import Bunch
from sklearn.datasets import fetch_species_distributions
from sklearn.datasets.species_distributions import construct_grids
from sklearn import svm, metrics


try:
    from mpl_toolkits.basemap import Basemap
    basemap = True
except ImportError:
    basemap = False

print(__doc__)

def create_species_bunch(species_name, train, test, coverages, xgrid, ygrid):
    bunch = Bunch(name=' '.join(species_name.split("_")[:2]))
    species_name = species_name.encode('ascii')
    points = dict(test=test, train=train)

    for label, pts in points.items():
        pts = pts[pts['species'] == species_name]
        bunch['pts_%s' % label] = pts

        ix = np.searchsorted(xgrid, pts['dd long'])
        iy = np.searchsorted(ygrid, pts['dd lat'])
        bunch['cov_%s' % label] = coverages[:, -iy, ix].T

    return bunch

def plot_species_distribution(species=("bradypus_variegatus_0", "microryzomys_minutus_0")):
    if len(species) > 2: 
        print("Note: when more than two species are provided," " only the first two will be used")

    t0 = time()

    data = fetch_species_distributions()

    xgrid,ygrid = construct_grids(data)

    X, Y = np.meshgrid(xgrid, ygrid[::1])

    BV_bunch = create_species_bunch(species[0],data.train, data.test, data.coverages, xgrid, ygrid)

    MM_bunch = create_species_bunch(species[1], data.train, data.test, data.coverages, xgrid, ygrid)

    background_points = np.c_[np.random.randint(low=0, high=data.Ny, size=10000),
                              np.random.randint(low=0, high=data.Nx, size=10000)].T

    land_reference = data.coverages[6]

    for i, species in enumerate([BV_bunch, MM_bunch]):
        print("_" * 80)
        print("Modeing distributions of species '%s'" % species.name)

        mean = species.cov_train.mean(axis=0)
        std= species.cov_train.std(axis=0)
        train_cover_std = (species.cov_train - mean) / std

        print(" - fit OneClassSVM ... ", end='')
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.5)
        clf.fit(train_cover_std)
        print("done.")

        plt.subplot(1,2, i+ 1) 
        if basemap:
            print(" - plt coastlines using basemap")
            m = Basemap(projection='cyl', llcrnrlat=Y.min(),
                        urcrnrlat=Y.max(), llcrnlon=X.min(),
                        urcrnrlon=X.max(), resolution='c')
            m.drawcoastlines()
            m.drawcountries()
        else:
            print(" - plot coastlines from coverage")
            plt.contour(X, Y, land_reference,
                        levels=[-9999], colors="k",
                        linestyles="solid")
            plt.xticks([])
            plt.yticks([])

        print(" - predict species distribtions")

        Z = np.ones((data.Ny, data.Nx), dtype=np.float64)

        idx = np.where(land_reference > - 9999)
        coverages_land = data.coverages[:, idx[0], idx[1]].T

        pred = clf.decision_function((coverages_land - mean) / std)[:, 0]
        Z *= pred.min()
        Z[idx[0], idx[1]] = pred

        levels = np.linspace(Z.min(), Z.max(), 25)
        Z[land_reference == -9999] = -9999

        plt.contour(X, Y, Z, levels=levels, cmap=plt.cm.Reds)
        plt.colorbar(format="%.2F")

        plt.scatter(species.pts_train['dd long'], species.pts_train['dd lat'],
                    s=2 ** 2, c='black',
                    marker='^', label="train")
        plt.scatter(species.pts_test['dd long'], species.pts_test['dd lat'],
                    marker='x', label="test")
        plt.legend()
        plt.title(species.name)
        plt.axis("equal")

        pred_background = Z[background_points[0], background_points[1]]
        pred_test = clf.decision_function((species.cov_test - mean)
                                         / std)[:, 0]

        scores = np.r_[pred_test, pred_background]
        y = np.r_[np.ones(pred_test.shape), np.zeros(pred_background.shape)]
        fpr, tpr, thresholds = metrics.roc_curve(y, scores)
        roc_auc = metrics.auc(fpr, tpr)
        plt.text(-35, -70, "AUC: %3f" % roc_auc, ha="right")
        print("\n Area under the ROC curve: %f" % roc_auc)

    print("\ntime elapsed: %.2fs" % (time() - t0))

plot_species_distribution()
plt.show()
