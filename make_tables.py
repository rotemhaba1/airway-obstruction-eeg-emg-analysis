"""Summarize the predictions in results/ into the tables of the paper (written to results/tables/).

    python make_tables.py

Metrics are computed for each held-out participant. Unless stated otherwise, the AUC is averaged over the four
participants for each channel and then over the channels of a modality (EEG: T7, T8, T7+T8; EMG: four muscles);
with repeated calibration draws, this is done per draw and reported as mean and SD over draws.
"""
import argparse
from pathlib import Path

import pandas as pd

from airway.config import CHANNELS, EEG_CHANNELS, LEVELS
from airway.metrics import evaluate, fuse

KEYS = ["method", "draw", "case", "dr", "k", "channel", "subject"]
AUC_LEVELS = [f"auc_{lv}" for lv in LEVELS]


def per_subject(path):
    """Metrics of every run, channel and held-out participant, including the decision-level fusion (DLF)."""
    pred = pd.read_csv(path)
    for col, default in (("case", 0), ("dr", "lda"), ("k", 5)):
        if col not in pred:
            pred[col] = default
    single = pred[(pred["method"] == "proposed") & pred["channel"].isin(CHANNELS)]
    if len(single):
        pred = pd.concat([pred, fuse(single)], ignore_index=True)
    out = pd.DataFrame([{**dict(zip(KEYS, key)), **evaluate(g)} for key, g in pred.groupby(KEYS)])
    out["modality"] = out["channel"].map(
        lambda c: "DLF" if c == "DLF" else "EEG" if c in EEG_CHANNELS + ("T7+T8",) else "EMG")
    return out


def mean_sd(x, by, over, metrics=("auc",)):
    """Average over everything below `over`, then mean and SD over `over`."""
    g = x.groupby(by + [over])[list(metrics)].mean().groupby(by)
    return g.mean().round(3).add_suffix("_mean").join(g.std().round(3).add_suffix("_sd"))


def over_draws(m, by, metrics=("auc",)):
    per_channel = m.groupby(by + ["modality", "draw", "channel"])[list(metrics)].mean().reset_index()
    return mean_sd(per_channel, by + ["modality"], "draw", metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    res = Path(parser.parse_args().results)
    m = {f.stem: per_subject(f) for f in res.glob("*.csv")}
    tables = {}

    if "main" in m:  # calibration draw of Tables 2-4; SD across participants
        x = m["main"]
        tables["auc_per_channel"] = mean_sd(x, ["dr", "case", "channel"], "subject")
        tables["auc_per_modality"] = mean_sd(x[x["channel"] != "DLF"], ["dr", "case", "modality"], "subject")
        lda = x[x["dr"] == "lda"].groupby(["case", "modality", "channel"])[AUC_LEVELS].mean()
        tables["max_auc_per_level"] = lda.groupby(["case", "modality"]).max().round(3)

    if "draws" in m:  # 20 random draws
        x = m["draws"]
        tables["draws"] = over_draws(x, ["case"])
        metrics = ["auc", "balanced_accuracy", "macro_f1"]
        part = x.groupby(["case", "modality", "subject", "draw"])[metrics].mean().reset_index()
        overall = part.groupby(["case", "modality", "draw"])[metrics].mean().reset_index().assign(subject="all")
        part = pd.concat([part.astype({"subject": str}), overall])
        tables["participants"] = mean_sd(part, ["case", "modality", "subject"], "draw", metrics)

    if "label_budget" in m and "draws" in m:  # k = 2-5 labeled segments per level, Case 3
        x = pd.concat([m["label_budget"], m["draws"][m["draws"]["case"] == 3]])
        tables["label_budget"] = over_draws(x[x["channel"] != "DLF"], ["k"], ["auc"] + AUC_LEVELS)

    if "recording_separated" in m:
        tables["recording_separated"] = over_draws(m["recording_separated"], ["case"])

    if "baselines" in m:
        tables["baselines"] = over_draws(m["baselines"], ["method"])

    (res / "tables").mkdir(exist_ok=True)
    for name, t in tables.items():
        t.to_csv(res / "tables" / f"{name}.csv")
        print(f"\n== {name}\n{t.to_string()}")


if __name__ == "__main__":
    main()
