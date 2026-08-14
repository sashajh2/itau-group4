# Notebooks

| Folder | Contents |
|---|---|
| [`embedding_experiments/`](embedding_experiments/) | Per-model stratified evaluation notebooks, split by modality: `audio/` (HuBERT, wav2vec2, MFCC), `video/` (SENet, ArcFace, FaceNet, MagFace, MARLIN), `forensic/` |
| [`colab/`](colab/) | Google Colab notebooks for training and testing the time-series models on GPU |
| [`exploration/`](exploration/) | Ad-hoc exploratory notebooks: audio, video, and forensic embedding tests, `deepfake_realtrace.ipynb`, and `deepfake_embeddings_2.ipynb` (LSTM over the 3328-d concatenation of all three embeddings) |

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

## A note on `Ensemble_Setup.ipynb`

A second copy of this notebook used to sit at the repository root. It was a 6-cell
Colab skeleton and a strict subset of
[`embedding_experiments/Ensemble_Setup.ipynb`](embedding_experiments/Ensemble_Setup.ipynb)
(22 cells), which supersedes it: same threshold and data-splitting helpers, a newer
`LogisticModule` that takes `max_iters`, and the full ensemble pipeline — hard/soft
voting plus the random-forest ensemble — that the skeleton lacked entirely.

The skeleton was deleted during the repository reorganization. Recover it from git
history if needed:

```bash
git log --oneline --diff-filter=D -- notebooks/ensemble/Ensemble_Setup.ipynb
git show <commit>^:notebooks/ensemble/Ensemble_Setup.ipynb > Ensemble_Setup_old.ipynb
```
