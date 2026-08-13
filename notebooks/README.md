# Notebooks

| Folder | Contents |
|---|---|
| [`embedding_experiments/`](embedding_experiments/) | Per-model stratified evaluation notebooks, split by modality: `audio/` (HuBERT, wav2vec2, MFCC), `video/` (SENet, ArcFace, FaceNet, MagFace, MARLIN), `forensic/` |
| [`colab/`](colab/) | Google Colab notebooks for training and testing the time-series models on GPU |
| [`exploration/`](exploration/) | Ad-hoc exploratory notebooks: audio, video, and forensic embedding tests, plus `deepfake_realtrace.ipynb` |
| [`ensemble/`](ensemble/) | Ensemble setup notebook |

## Colab notebooks

The notebooks in `colab/` are written to run **outside this repository**. They mount
Google Drive and copy the model source file from a hardcoded Drive path such as
`/content/drive/MyDrive/MIT/Lab/time_series_model.py`, rather than importing from the
repo. To use them you need to upload the relevant file from `models/time_series/` to
Drive and update `DRIVE_MODEL_PATH` at the top of the notebook.

| Notebook | Model |
|---|---|
| `Colab_TimeSeries_Training.ipynb` / `Colab_TimeSeries_Testing.ipynb` | The PatchTST-style temporal Transformer (`time_series_model.py`) |
| `Colab_TimeSeriesCNN_Training.ipynb` / `Colab_TimeSeriesCNN_Testing.ipynb` | The 1D CNN (`time_series_cnn_model.py`), self-contained — does not need `time_series_model.py` |
| `Colab_Detailed_Testing.ipynb` | Runs `test_model_detailed.py` against a trained checkpoint |

Checkpoints are not interchangeable between the Transformer and the CNN.

## Two versions of `Ensemble_Setup.ipynb`

There are two files with this name and they have **different contents** — this is not a
duplicate:

- `ensemble/Ensemble_Setup.ipynb` — the copy that was at the repository root
- `embedding_experiments/Ensemble_Setup.ipynb` — the copy that was already in this folder

Neither was deleted, since it is not clear which supersedes the other. Diff them before
relying on either.
