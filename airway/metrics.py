"""Evaluation metrics and decision-level fusion."""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from .config import CHANNELS, LEVELS
from .models import labels

SCORE = [f"score_{lv}" for lv in LEVELS]
ENS = [f"ens_{lv}" for lv in LEVELS]


def auc_per_level(y, score):
    """One-vs-rest AUC of each obstruction level."""
    return {lv: roc_auc_score(y == lv, score[:, i]) for i, lv in enumerate(LEVELS)}


def evaluate(pred):
    """Metrics of one held-out participant. AUC: one-vs-rest AUC of each level, averaged with weights
    proportional to the number of test segments of each level. Labels: argmax of the ensemble probabilities."""
    y = pred["level"].to_numpy()
    per_level = auc_per_level(y, pred[SCORE].to_numpy())
    out = {"auc": sum(per_level[lv] * np.mean(y == lv) for lv in LEVELS)}
    out.update({f"auc_{lv}": v for lv, v in per_level.items()})
    if ENS[0] in pred:
        y_hat = labels(pred[ENS].to_numpy())
        out["balanced_accuracy"] = balanced_accuracy_score(y, y_hat)
        out["macro_f1"] = f1_score(y, y_hat, average="macro")
    return out


def fuse(pred, channels=CHANNELS):
    """Decision-level fusion: equal-weight mean of the probabilities of the given channels, for the test segments
    available in all of them (`pred` holds one row per channel and segment)."""
    p = pred[pred["channel"].isin(channels)]
    keys = [c for c in p.columns if c not in SCORE + ENS + ["channel", "level"]]
    n = p.groupby(keys)["channel"].transform("nunique")
    p = p[n == len(channels)]
    fused = p.groupby(keys)[SCORE + ["level"]].mean().reset_index()
    fused["channel"] = "DLF"
    return fused
