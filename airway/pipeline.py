"""Cross-subject classification of one held-out participant (Cases 1-4).

Case 1: each source participant separately, no alignment      Case 3: as Case 1, with affine alignment
Case 2: source participants merged, no alignment               Case 4: as Case 2, with affine alignment

The held-out participant contributes k labeled calibration segments per obstruction level. They are used to fit
that participant's dimensionality reduction (LDA) and, in Cases 3 and 4, the affine alignment, and they are
excluded from the test set. For every anchor (source) participant, the class probabilities are computed, and the
probabilities are averaged over the three anchors.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import LEVELS, N_CALIBRATION, SUBJECTS, modality
from .data import calibration_rows, load_channel
from .models import Affine, classify, reducer

CASES = {1: (False, False), 2: (True, False), 3: (False, True), 4: (True, True)}  # (sources merged, aligned)


def first_k(rows, level, k):
    """First k rows of each obstruction level, in the given order."""
    count, out = {}, []
    for r in rows:
        if count.get(level[r], 0) < k:
            out.append(r)
            count[level[r]] = count.get(level[r], 0) + 1
    return out


def test_rows(meta, target, calib, recording_separated=False):
    """Test segments of the held-out participant: all segments except the calibration segments (and, in the
    recording-separated evaluation, except every segment of the recordings that contain calibration segments)."""
    own = meta.index[meta["subject"] == target]
    cal = [r for r in calib if meta.at[r, "subject"] == target]
    drop = set(cal)
    if recording_separated:
        drop = set(own[meta.loc[own, "record"].isin(meta.loc[cal, "record"])])
    return np.array([r for r in own if r not in drop])


def embed(x, meta, target, anchor, case, calib, k=N_CALIBRATION, dr="lda"):
    """Low-dimensional (and, in Cases 3-4, aligned) representation of the training and held-out participants."""
    merged, aligned = CASES[case]
    subject, level = meta["subject"].to_numpy(), meta["level"].to_numpy()
    rows = np.flatnonzero(np.ones(len(meta), bool) if merged else np.isin(subject, [target, anchor]))
    z = pd.DataFrame(StandardScaler().fit_transform(x[rows]), index=rows)  # label-free scaling of all rows used

    cal_target = first_k([r for r in calib if subject[r] == target], level, k)
    emb = {}
    for s in pd.unique(subject[rows]):
        own = rows[subject[rows] == s]
        fit_rows = own if (s != target or dr == "pca") else cal_target  # PCA uses no labels
        model = reducer(dr).fit(z.loc[fit_rows], level[fit_rows] if dr == "lda" else None)
        emb[s] = pd.DataFrame(model.transform(z.loc[own]), index=own)

    if aligned:
        ref = emb[anchor]
        cal_anchor = [r for r in calib if subject[r] == anchor]
        for s in emb:
            if s == anchor:
                continue
            if s == target:  # calibration segments of the held-out participant -> those of the anchor
                f = Affine().fit(ref.loc[cal_anchor].to_numpy(), level[cal_anchor],
                                 emb[s].loc[cal_target].to_numpy(), level[cal_target])
            else:  # other source participants -> anchor, all segments
                f = Affine().fit(ref.to_numpy(), level[ref.index], emb[s].to_numpy(), level[emb[s].index])
            emb[s] = pd.DataFrame(f.transform(emb[s].to_numpy()), index=emb[s].index)

    train = pd.concat([emb[s] for s in emb if s != target]).sort_index()
    return train, emb[target]


def run(channels, target, case, draw, data_dir, splits, k=N_CALIBRATION, dr="lda", recording_separated=False):
    """Class probabilities for the test segments of one held-out participant.

    `channels` is one channel or the pair ("T7", "T8"), whose low-dimensional outputs are concatenated.
    Returns one row per test segment with the RBF-SVM and ensemble probabilities and the true level.
    """
    data = [load_channel(data_dir, c) for c in channels]
    calibs = [calibration_rows(splits, draw, c, m) for c, (_, m) in zip(channels, data)]
    meta = data[0][1]
    test = test_rows(meta, target, calibs[0], recording_separated)
    for (_, m), cal in zip(data[1:], calibs[1:]):
        if not m["segment"].equals(meta["segment"]):
            raise ValueError("channels classified together must contain the same segments")
        test = np.intersect1d(test, test_rows(m, target, cal, recording_separated))

    anchors = [s for s in SUBJECTS if s != target]
    if case == 2:
        anchors = anchors[:1]  # merged sources without alignment do not depend on the anchor
    p_rbf, p_ens = [], []
    for anchor in anchors:
        parts = [embed(x, m, target, anchor, case, cal, k, dr) for (x, m), cal in zip(data, calibs)]
        train = pd.concat([p[0] for p in parts], axis=1)
        held = pd.concat([p[1] for p in parts], axis=1)
        a, b = classify(train.to_numpy(), meta.loc[train.index, "level"].to_numpy(),
                        held.loc[test].to_numpy(), modality(channels[0]))
        p_rbf.append(a)
        p_ens.append(b)
    return result_frame(meta.loc[test], np.mean(p_rbf, axis=0), np.mean(p_ens, axis=0))


def result_frame(meta, score, p_ens=None):
    """score_*: values used for the AUC (RBF-SVM probabilities unless stated); ens_*: ensemble probabilities,
    used for the predicted labels."""
    out = pd.DataFrame({"segment": meta["segment"].to_numpy(), "level": meta["level"].to_numpy()})
    for i, lv in enumerate(LEVELS):
        out[f"score_{lv}"] = score[:, i]
        if p_ens is not None:
            out[f"ens_{lv}"] = p_ens[:, i]
    return out
