"""Run the analyses and save the per-segment predictions to results/<experiment>.csv.

    python run_experiments.py main                  # calibration draw of Tables 2-4: Cases 1-4, LDA and PCA
    python run_experiments.py draws                 # 20 random calibration draws, Cases 3-4
    python run_experiments.py label_budget          # k = 2, 3, 4 labeled segments per level, 20 draws, Case 3
    python run_experiments.py recording_separated   # 10 recording-separated draws, Cases 3-4
    python run_experiments.py baselines             # zero-shot, common-space and target-only baselines, 20 draws
"""
import argparse
import itertools
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from airway import baselines, pipeline
from airway.config import CHANNELS, SUBJECTS
from airway.data import read_draws

CHANNEL_SETS = [(c,) for c in CHANNELS] + [("T7", "T8")]


def jobs(experiment, splits):
    """(function, keyword arguments, labels) for every unit of work."""
    draws = sorted(read_draws(splits)["draw"].unique())
    random_draws = [d for d in draws if d.startswith("d")]
    block_draws = [d for d in draws if d.startswith("blk")]
    if experiment == "main":
        grid = itertools.product(["main"], [1, 2, 3, 4], ["lda", "pca"], [5], CHANNEL_SETS, SUBJECTS, [False])
    elif experiment == "draws":
        grid = itertools.product(random_draws, [3, 4], ["lda"], [5], CHANNEL_SETS, SUBJECTS, [False])
    elif experiment == "label_budget":
        grid = itertools.product(random_draws, [3], ["lda"], [2, 3, 4], CHANNEL_SETS, SUBJECTS, [False])
    elif experiment == "recording_separated":
        grid = itertools.product(block_draws, [3, 4], ["lda"], [5], CHANNEL_SETS, SUBJECTS, [True])
    elif experiment == "baselines":
        for method, draw, channel, s in itertools.product(baselines.METHODS, random_draws, CHANNELS, SUBJECTS):
            yield baselines.run, dict(method=method, channel=channel, target=s, draw=draw), \
                dict(method=method, draw=draw, channel=channel, subject=s, k=5)
        return
    else:
        raise ValueError(experiment)
    for draw, case, dr, k, chans, s, sep in grid:
        yield pipeline.run, dict(channels=chans, target=s, case=case, draw=draw, k=k, dr=dr, recording_separated=sep), \
            dict(method="proposed", draw=draw, case=case, dr=dr, k=k, channel="+".join(chans), subject=s)


def unit(func, kwargs, labels, data_dir, splits):
    out = func(data_dir=data_dir, splits=splits, **kwargs)
    for key, value in reversed(list(labels.items())):
        out.insert(0, key, value)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["main", "draws", "label_budget", "recording_separated", "baselines"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--splits", default="splits/calibration_draws.csv")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--jobs", type=int, default=1, help="parallel workers")
    args = parser.parse_args()

    work = list(jobs(args.experiment, args.splits))
    parts = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(unit)(f, kw, lab, args.data_dir, args.splits) for f, kw, lab in work)
    Path(args.out_dir).mkdir(exist_ok=True)
    out = Path(args.out_dir) / f"{args.experiment}.csv"
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
