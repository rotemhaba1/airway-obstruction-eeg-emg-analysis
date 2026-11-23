# Few-shot cross-subject classification of simulated tracheal obstruction levels from surface EMG and EEG

Analysis code for the paper *Few-Shot Cross-Subject Classification of Simulated Tracheal Obstruction Levels from
Surface EMG and EEG: A Feasibility Study* (R. Haba, G. Singer, S. Naftali, S. Torick, D. Adler, A. Ratnovsky).

Four healthy participants breathed through tubes simulating 0%, 20%, 40% and 60% tracheal obstruction while EEG
(T7, T8) and surface EMG (sternocleidomastoid and external intercostal muscles, both sides) were recorded.
The obstruction level of each inspiratory segment of a held-out participant is classified from the data of the
other three participants and five labeled calibration segments per level of the held-out participant.

## Method

For every held-out participant and channel:

1. **Features.** STFT magnitude of each segment (EEG: 20-sample Hamming window, 40% overlap; EMG: 300-sample
   Tukey window, 50% overlap; PSD scaling), flattened to a vector.
2. **Scaling.** z-scoring of all segments used in the run (no labels).
3. **Dimensionality reduction.** A 3-D LDA fitted separately for each participant: on all labeled segments of a
   source participant, and on the 20 calibration segments of the held-out participant (2-D PCA as an alternative).
4. **Alignment** (Cases 3 and 4). A least-squares affine map from the held-out participant to an anchor source
   participant, estimated from class-matched calibration segments. In Case 4 the other source participants are also
   aligned to the anchor.
5. **Classification.** Soft-voting ensemble of k-NN (k = 21), linear SVM and RBF SVM (C = 1, gamma = 2), trained
   on the source participants. Probabilities are averaged over the three anchors. The AUC is computed from the
   RBF-SVM probabilities and the predicted labels from the ensemble.

| Case | Source participants | Alignment |
|---|---|---|
| 1 | each separately | none |
| 2 | merged | none |
| 3 | each separately | affine |
| 4 | merged, aligned to a common anchor | affine |

The calibration segments are excluded from the test set. Decision-level fusion (DLF) is the equal-weight mean of
the RBF-SVM probabilities of the six channels, on the test segments common to all of them. The classifier
hyperparameters and the ensemble and fusion weights were fixed in advance; the labels of the test segments are
used only for evaluation.

Baselines (same calibration and test segments): zero-shot (scaler, LDA and classifiers fitted on the source
participants only), common-space alignment (per-participant z-scoring, CORAL, class-centroid, Procrustes and affine
alignment, nearest class centroid, source-plus-target training) and a target-only RBF SVM trained on the 20
calibration segments.

## Repository structure

```
airway/                 analysis package
    config.py           channels, STFT parameters, classifier settings
    data.py             segments, STFT features and calibration draws
    models.py           LDA/PCA, affine alignment, classifiers
    pipeline.py         Cases 1-4 for one held-out participant
    baselines.py        zero-shot, common-space and target-only baselines
    metrics.py          AUC, balanced accuracy, macro-F1, decision-level fusion
data/                   recordings (not included, see data/README.md)
splits/                 calibration draws
run_experiments.py      runs the analyses and saves per-segment predictions
make_tables.py          summarizes the predictions
```

## Calibration draws

`splits/calibration_draws.csv` lists the calibration segments of every draw (columns: draw, channel, participant,
level, rank, segment identifier):

- `main`: the draw used for Tables 2-4 of the paper;
- `d01`-`d20`: 20 random draws, the same breaths for all channels;
- `blk01`-`blk10`: recording-separated draws, in which the calibration segments of each level come from one
  recording and all segments of that recording are removed from the test set.

The first *k* segments of each level (by rank) form the calibration set for a label budget of *k*.

## Usage

Python 3.9 was used.

```
pip install -r requirements.txt
python run_experiments.py main
python run_experiments.py draws
python run_experiments.py label_budget
python run_experiments.py recording_separated
python run_experiments.py baselines
python make_tables.py
```

`run_experiments.py` writes one CSV per analysis to `results/`, with one row per test segment (`score_*`: values
used for the AUC; `ens_*`: ensemble probabilities). Use `--jobs N` to run in parallel. `make_tables.py` computes the
metrics for every held-out participant and writes the summary tables to `results/tables/`.

## License

MIT, see `LICENSE`.
