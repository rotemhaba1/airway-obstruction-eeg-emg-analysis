# Data

The recordings are not included in this repository (see the Data Availability statement of the paper).
To run the analyses, place the segmented recordings in this folder, one CSV file per channel:

| File | Signal |
|---|---|
| `T7.csv`, `T8.csv` | EEG, temporal channels (128 Hz) |
| `STr.csv`, `STl.csv` | surface EMG, right and left sternocleidomastoid (1 kHz) |
| `EIr.csv`, `EIl.csv` | surface EMG, right and left external intercostal (1 kHz) |

Each file has one header row and one column per inspiratory segment. The signal samples of the segment are
followed by four rows:

1. recording index
2. obstruction level (0, 20, 40 or 60)
3. participant (1-4)
4. segment identifier (unique within a file; the same breath has the same identifier in all files)

All segments of a file have the same length (shorter segments are extended with the subsequent recorded samples).
The segment identifiers are those used in `splits/calibration_draws.csv`.
