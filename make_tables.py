"""Summarize the predictions in results/ into the tables reported in the paper (written to results/tables/).

    python make_tables.py

AUC values are computed for each held-out participant and averaged over the four participants; with repeated
calibration draws, they are computed per draw and reported as mean +/- SD over draws (sample SD).
"""
import argparse
from pathlib import Path

import pandas as pd

from airway.config import CHANNELS, EEG_CHANNELS, EMG_CHANNELS, LEVELS
from airway.metrics import evaluate, fuse

MODALITY = {**{c: "EEG" for c in EEG_CHANNELS + ("T7+T8",)}, **{c: "EMG" for c in EMG_CHANNELS}, "DLF": "DLF"}
KEYS = ["method", "draw", "case", "dr", "k", "channel", "subject"]


def per_subject(pred):
    """Metrics for every held-out participant, channel and run; adds the decision-level fusion of the six channels."""
    pred = pred.copy()
    for col in ("case", "dr", "k"):
        if col not in pred:
            pred[col] = {"case": 0, "dr": "lda", "k": 5}[col]
    single = pred[pred["channel"].isin(CHANNELS) & (pred["method"] == "proposed")]
    if len(single):
        pred = pd.concat([pred, fuse(single)], ignore_index=True)
    rows = [{**dict(zip(KEYS, key)), **evaluate(g)} for key, g in pred.groupby(KEYS)]
    out = pd.DataFrame(rows)
    out["modality"] = out["channel"].map(MODALITY)
    return out


def mean_sd(x, by, level, metrics=("auc",)):
    """Mean over the levels below `level`, then mean and SD over `level`."""
    g = x.groupby(by + [level])[list(metrics)].mean().groupby(by)
    return g.mean().round(3).add_suffix("_mean").join(g.std().round(3).add_suffix("_sd"))


def over_draws(m, by):
    """AUC per draw: mean over participants for each channel, then over the channels of a modality."""
    per_channel = m.groupby(by + ["modality", "draw", "channel"])["auc"].mean().reset_index()
    return mean_sd(per_channel, by + ["modality"], "draw")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    res, out = Path(args.results), Path(args.results) / "tables"
    out.mkdir(exist_ok=True)

    def load(name):
        f = res / f"{name}.csv"
        return per_subject(pd.read_csv(f)) if f.exists() else None

    tables = {}
    m = load("main")
    if m is not None:  # calibration draw of Tables 2-4
        tables["auc_per_channel"] = mean_sd(m, ["dr", "case", "channel"], "subject")
        mod = m[m["channel"] != "DLF"].groupby(["dr", "case", "modality", "subject"])["auc"].mean().reset_index()
        tables["auc_per_modality"] = mean_sd(mod, ["dr", "case", "modality"], "subject")
        per_level = m[m["dr"] == "lda"].groupby(["case", "modality", "channel"])[[f"auc_{lv}" for lv in LEVELS]].mean()
        tables["max_auc_per_level"] = per_level.groupby(["case", "modality"]).max().round(3)

    d = load("draws")
    if d is not None:  # 20 random calibration draws
        tables["draws_summary"] = over_draws(d, ["case"])
        metrics = ["auc", "balanced_accuracy", "macro_f1"]
        part = d[d["channel"] != "DLF"].groupby(["case", "modality", "subject", "draw"])[metrics].mean().reset_index()
        overall = part.groupby(["case", "modality", "draw"])[metrics].mean().reset_index().assign(subject="all")
        part = pd.concat([part.astype({"subject": str}), overall])
        tables["participants"] = mean_sd(part, ["case", "modality", "subject"], "draw", metrics)
        dlf = d[d["channel"] == "DLF"].astype({"subject": str})
        dlf = pd.concat([dlf, dlf.groupby(["case", "draw"])["auc"].mean().reset_index().assign(subject="all")])
        tables["dlf_participants"] = mean_sd(dlf, ["case", "subject"], "draw")

    k = load("label_budget")
    if k is not None and d is not None:  # k = 2-5 per level; k = 5 from the 20 draws (Case 3)
        both = pd.concat([k, d[d["case"] == 3]])
        both = both[both["channel"] != "DLF"]
        per_channel = both.groupby(["k", "modality", "draw", "channel"])[
            ["auc"] + [f"auc_{lv}" for lv in LEVELS]].mean().reset_index()
        tables["label_budget"] = mean_sd(per_channel, ["k", "modality"], "draw",
                                         ["auc"] + [f"auc_{lv}" for lv in LEVELS])

    r = load("recording_separated")
    if r is not None:
        tables["recording_separated"] = over_draws(r, ["case"])

    b = load("baselines")
    if b is not None:
        b["modality"] = b["channel"].map(MODALITY)
        tables["baselines"] = over_draws(b, ["method"])

    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv")
        print(f"\n== {name}\n{t.to_string()}")


if __name__ == "__main__":
    main()
