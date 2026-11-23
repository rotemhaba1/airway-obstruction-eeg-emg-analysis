"""Time-frequency features."""
import numpy as np
from scipy import signal

from .config import SAMPLING_RATE, STFT, modality


def stft_features(signals, channel):
    """STFT magnitude of each segment (rows of `signals`), flattened to (frequency bins x time frames)."""
    mod = modality(channel)
    p = STFT[mod]
    n = p["nperseg"]
    _, _, z = signal.stft(signals, fs=SAMPLING_RATE[mod], window=signal.get_window(p["window"], n),
                          nperseg=n, noverlap=int(p["overlap"] * n), scaling="psd", axis=-1)
    return np.abs(z).reshape(len(signals), -1)
