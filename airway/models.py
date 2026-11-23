"""Building blocks: dimensionality reduction, affine alignment and classifiers."""
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .config import ENSEMBLE_WEIGHTS, KNN_NEIGHBORS, LEVELS, RBF_GAMMA, SEED, SVM_C


def reducer(method):
    """3-D LDA (supervised) or 2-D PCA (unsupervised)."""
    if method == "lda":
        return LinearDiscriminantAnalysis(n_components=3)
    if method == "pca":
        return PCA(n_components=2)
    raise ValueError(method)


class Affine:
    """Least-squares affine map x -> x R + t, estimated from class-matched pairs of points.

    Within each class, the i-th point of the source set is paired with the i-th point of the target set.
    """

    def fit(self, target, y_target, source, y_source):
        p, q = [], []
        for level in pd.unique(y_target):
            a, b = target[y_target == level], source[y_source == level]
            n = min(len(a), len(b))
            p.append(a[:n])
            q.append(b[:n])
        q = np.vstack(q)
        coef = linalg.lstsq(np.hstack([q, np.ones((len(q), 1))]), np.vstack(p))[0]
        self.R, self.t = coef[:-1], coef[-1]
        return self

    def transform(self, x):
        return x @ self.R + self.t


def rbf_svm():
    return SVC(kernel="rbf", C=SVM_C, gamma=RBF_GAMMA, probability=True)


def ensemble(modality):
    members = [("knn", KNeighborsClassifier(KNN_NEIGHBORS, weights="distance", metric="l2", algorithm="ball_tree")),
               ("svm", SVC(kernel="linear", C=SVM_C, probability=True)),
               ("rbf", rbf_svm())]
    return VotingClassifier(members, voting="soft", weights=ENSEMBLE_WEIGHTS[modality])


def classify(x_train, y_train, x_test, modality):
    """Class probabilities of the RBF SVM (used for the AUC) and of the soft-voting ensemble (used for the labels).

    The global seed is set before each fit so that the internal cross-validation of the Platt scaling is fixed.
    """
    np.random.seed(SEED)
    p_ens = ensemble(modality).fit(x_train, y_train).predict_proba(x_test)
    np.random.seed(SEED)
    p_rbf = rbf_svm().fit(x_train, y_train).predict_proba(x_test)
    return p_rbf, p_ens


def labels(prob):
    return np.asarray(LEVELS)[np.argmax(prob, axis=1)]
