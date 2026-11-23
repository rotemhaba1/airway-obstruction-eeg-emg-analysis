"""Baselines evaluated on the same calibration and test segments as the proposed method.

All except `target_only` work in a common space: a scaler and a 3-D LDA fitted on the three source participants
only, followed by the classifiers of the proposed method trained on the source participants.
"""
import numpy as np
from scipy.linalg import inv, orthogonal_procrustes, sqrtm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import LEVELS, N_CALIBRATION, RBF_GAMMA, SEED, SVM_C, modality
from .data import calibration_rows, load_channel
from .models import Affine, classify, reducer
from .pipeline import first_k, result_frame, test_rows

METHODS = ("zero_shot", "zscore", "coral", "source_plus_target", "centroid", "procrustes", "affine",
           "nearest_centroid", "target_only")


def centroids(z, y):
    return np.stack([z[y == lv].mean(axis=0) for lv in LEVELS])


def run(method, channels, target, draw, data_dir, splits, k=N_CALIBRATION, recording_separated=False):
    """`channels`: one channel, or ("T7", "T8"), whose features are concatenated."""
    data = [load_channel(data_dir, c) for c in channels]
    if any(not m["segment"].equals(data[0][1]["segment"]) for _, m in data):
        raise ValueError("channels classified together must contain the same segments")
    x, meta = np.hstack([d[0] for d in data]), data[0][1]
    calib = calibration_rows(splits, draw, channels[0], meta)
    subject, y = meta["subject"].to_numpy(), meta["level"].to_numpy()
    test = test_rows(meta, target, calib, recording_separated)
    cal = np.array(first_k([r for r in calib if subject[r] == target], y, k))
    src = subject != target
    mod = modality(channels[0])

    if method == "target_only":  # RBF SVM of the proposed method, trained on the calibration segments only
        sc = StandardScaler().fit(x[cal])
        lda = reducer("lda").fit(sc.transform(x[cal]), y[cal])
        svm = SVC(kernel="rbf", C=SVM_C, gamma=RBF_GAMMA, probability=True, random_state=SEED)
        svm.fit(lda.transform(sc.transform(x[cal])), y[cal])
        # decision values: Platt probabilities are unreliable with 20 training segments
        return result_frame(meta.loc[test], svm.decision_function(lda.transform(sc.transform(x[test]))))

    if method == "source_plus_target":
        rows = np.concatenate([np.flatnonzero(src), cal])
        sc = StandardScaler().fit(x[rows])
        lda = LinearDiscriminantAnalysis(n_components=3).fit(sc.transform(x[rows]), y[rows])
        z = lda.transform(sc.transform(x[rows]))
        return result_frame(meta.loc[test], *classify(z, y[rows], lda.transform(sc.transform(x[test])), mod))

    sc = StandardScaler().fit(x[src])
    lda = LinearDiscriminantAnalysis(n_components=3).fit(sc.transform(x[src]), y[src])
    zs, ys = lda.transform(sc.transform(x[src])), y[src]
    zt = lda.transform(sc.transform(x[test]))
    zc, yc = lda.transform(sc.transform(x[cal])), y[cal]

    if method == "zscore":  # each participant standardized with its own (unlabeled) data
        zt = lda.transform(StandardScaler().fit(x[subject == target]).transform(x[test]))
    elif method == "coral":  # second-order alignment with the unlabeled data of the held-out participant
        za = lda.transform(sc.transform(x[subject == target]))
        cs, ct = np.cov(zs.T) + 1e-6 * np.eye(3), np.cov(za.T) + 1e-6 * np.eye(3)
        zt = (zt - za.mean(axis=0)) @ np.real(inv(sqrtm(ct)) @ sqrtm(cs)) + zs.mean(axis=0)
    elif method == "centroid":
        zt = zt + centroids(zs, ys).mean(axis=0) - centroids(zc, yc).mean(axis=0)
    elif method == "procrustes":
        a, b = centroids(zc, yc), centroids(zs, ys)
        rot, s = orthogonal_procrustes(a - a.mean(0), b - b.mean(0))
        zt = (zt - a.mean(0)) @ rot * (s / max(np.sum((a - a.mean(0)) ** 2), 1e-12)) + b.mean(0)
    elif method == "affine":  # calibration segments -> class centroids of the source participants
        target_points = centroids(zs, ys)[[LEVELS.index(v) for v in yc]]
        zt = Affine().fit(target_points, yc, zc, yc).transform(zt)
    elif method == "nearest_centroid":
        d2 = ((zt[:, None, :] - centroids(zc, yc)[None]) ** 2).sum(axis=-1)
        return result_frame(meta.loc[test], -d2)
    elif method != "zero_shot":
        raise ValueError(method)
    return result_frame(meta.loc[test], *classify(zs, ys, zt, mod))
