"""Loading of the segmented recordings and of the calibration draws."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import N_CALIBRATION
from .features import stft_features

META = ["record", "level", "subject", "segment"]


def read_segments(path):
    """Read one segmented channel.

    File layout (CSV, one header row): one column per segment; the signal samples are followed by four rows
    holding the recording index, the obstruction level, the participant and the segment identifier.
    Segments are returned ordered by participant (original order kept within a participant).
    """
    raw = pd.read_csv(path, header=None, skiprows=1).to_numpy(dtype=float)
    signals, meta = raw[:-4].T, pd.DataFrame(raw[-4:].T.astype(int), columns=META)
    order = np.argsort(meta["subject"].to_numpy(), kind="stable")
    return signals[order], meta.iloc[order].reset_index(drop=True)


@lru_cache(maxsize=None)
def load_channel(data_dir, channel):
    """STFT features and metadata of one channel (cached)."""
    signals, meta = read_segments(Path(data_dir) / f"{channel}.csv")
    return stft_features(signals, channel), meta


@lru_cache(maxsize=None)
def _draws(path):
    return pd.read_csv(path)


def calibration_rows(splits_path, draw, channel, meta):
    """Row positions of the calibration segments of a draw, ordered by level, participant and rank."""
    d = _draws(str(splits_path))
    d = d[(d["draw"] == draw) & (d["channel"] == channel)].sort_values(["level", "subject", "rank"])
    if len(d) != N_CALIBRATION * 4 * meta["subject"].nunique():
        raise ValueError(f"incomplete draw {draw} for {channel}")
    row_of = pd.Series(meta.index, index=meta["segment"])
    return row_of.loc[d["segment"]].to_numpy()


def draw_names(splits_path, prefix):
    d = _draws(str(splits_path))
    return sorted(n for n in d["draw"].unique() if n.startswith(prefix))
