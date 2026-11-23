"""Segmented recordings, STFT features and calibration draws."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from .config import N_CALIBRATION, SAMPLING_RATE, STFT, modality


def read_segments(path):
    """One segmented channel (format described in data/README.md), ordered by participant."""
    raw = pd.read_csv(path, header=None, skiprows=1).to_numpy(dtype=float)
    meta = pd.DataFrame(raw[-4:].T.astype(int), columns=["record", "level", "subject", "segment"])
    order = np.argsort(meta["subject"].to_numpy(), kind="stable")
    return raw[:-4].T[order], meta.iloc[order].reset_index(drop=True)


def stft_features(signals, channel):
    """STFT magnitude of each segment, flattened to (frequency bins x time frames)."""
    p, fs = STFT[modality(channel)], SAMPLING_RATE[modality(channel)]
    n = p["nperseg"]
    _, _, z = signal.stft(signals, fs=fs, window=signal.get_window(p["window"], n), nperseg=n,
                          noverlap=int(p["overlap"] * n), scaling="psd", axis=-1)
    return np.abs(z).reshape(len(signals), -1)


@lru_cache(maxsize=None)
def load_channel(data_dir, channel):
    """Features and metadata of one channel (cached)."""
    signals, meta = read_segments(Path(data_dir) / f"{channel}.csv")
    return stft_features(signals, channel), meta


@lru_cache(maxsize=None)
def read_draws(path):
    return pd.read_csv(path)


def calibration_rows(splits, draw, channel, meta):
    """Row positions of the calibration segments of a draw, ordered by level, participant and rank."""
    d = read_draws(str(splits))
    d = d[(d["draw"] == draw) & (d["channel"] == channel)].sort_values(["level", "subject", "rank"])
    if len(d) != N_CALIBRATION * 4 * meta["subject"].nunique():
        raise ValueError(f"incomplete draw {draw} for {channel}")
    return pd.Series(meta.index, index=meta["segment"]).loc[d["segment"]].to_numpy()
