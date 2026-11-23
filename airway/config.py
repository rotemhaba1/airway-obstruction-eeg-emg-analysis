"""Settings used for all analyses reported in the paper."""

SUBJECTS = (1, 2, 3, 4)
LEVELS = (0, 20, 40, 60)  # obstruction level: % reduction of the central inner diameter of the tube

EEG_CHANNELS = ("T7", "T8")
EMG_CHANNELS = ("STr", "STl", "EIr", "EIl")  # sternocleidomastoid and external intercostal, right/left
CHANNELS = EEG_CHANNELS + EMG_CHANNELS

SAMPLING_RATE = {"EEG": 128, "EMG": 1000}

# STFT magnitude, PSD scaling, mean not removed
STFT = {
    "EEG": {"window": "hamming", "nperseg": 20, "overlap": 0.4},
    "EMG": {"window": "tukey", "nperseg": 300, "overlap": 0.5},
}

N_CALIBRATION = 5  # labeled calibration segments per obstruction level of the held-out participant

# Soft-voting ensemble (k-NN, linear SVM, RBF SVM). Hyperparameters and weights were fixed in advance.
KNN_NEIGHBORS = 21
SVM_C = 1.0
RBF_GAMMA = 2.0
ENSEMBLE_WEIGHTS = {"EEG": (1, 2, 1), "EMG": (2, 1, 1)}

SEED = 42


def modality(channel):
    return "EEG" if channel in EEG_CHANNELS else "EMG"
