# Documentation index

Start here. This file is a map of the repository: what each directory holds, which
documents to read in what order, and where the runnable entry points are.

Everything under `docs/` is prose. No code reads from this directory, so nothing here
can break by being moved or renamed.

---

## What this project is

Research code for **audiovisual deepfake detection**. The general shape of the work:

```
raw video datasets  →  segments  →  pretrained embeddings  →  downstream models
 (AVDeepfake1M,        (audio +      (HuBERT, OpenL3,          (linear probes, MLP
  ShareVeo3, DFD,       video         wav2vec2, MFCC /          classifiers, temporal
  MovieGen, SynVTA,     frames)       SENet, ArcFace,           transformers + LSTMs,
  Sora2)                              FaceNet, MagFace,         disentangled
                                      MARLIN / ResNet)          representation learning)
```

Embeddings are computed once and cached (as `.npy`/HDF5 files, in a Neon Postgres
pgvector store, and on Dropbox), then reused by every downstream experiment. Most
experiments therefore start from precomputed embeddings rather than from video.

---

## Reading order for someone new

1. [`architecture/BRANCH_DOCUMENTATION.md`](architecture/BRANCH_DOCUMENTATION.md) —
   the broadest overview of the detection pipeline and the model/architecture decisions
   behind it. **Read this first.**
2. [`analysis/experiment_summary.md`](analysis/experiment_summary.md) — what has actually
   been tried so far, phase by phase, and how each attempt turned out.
3. [`analysis/local_global_embedding_analysis.md`](analysis/local_global_embedding_analysis.md) —
   the most complete write-up of findings on the embedding space. (Marked as a first
   draft; citations in it are placeholders and are not yet verified.)
4. [`guides/PIPELINE_USAGE.md`](guides/PIPELINE_USAGE.md) — how to actually run the
   training/evaluation pipeline.
5. [`planning/ideas/`](planning/ideas/) — the open-questions backlog: what would be worth
   trying next.

---

## Directory map of `docs/`

| Folder | Holds |
|---|---|
| [`architecture/`](architecture/) | How the system and the models are built |
| [`guides/`](guides/) | How to run things — usage instructions |
| [`analysis/`](analysis/) | Experimental findings and interpretations of results |
| [`planning/`](planning/) | Plans, proposals, and the open-ideas backlog |
| [`refactoring/`](refactoring/) | Records of past restructures — historical context |
| [`slides/`](slides/) | Presentation decks (Marp format) |
| [`reference/`](reference/) | Background paper and external prompt material |

Each folder has its own `README.md` describing its contents file by file.

### Docs that live outside `docs/`

Package-level `README.md` files stay next to the code they describe, which is where a
Python developer expects them:

- [`../data/README.md`](../data/README.md) — the data package: loaders, preprocessing,
  embedding generation, storage
- [`../data/loaders/README.md`](../data/loaders/README.md) — per-dataset loaders
- [`../data/loaders/instagram/README_instagram.md`](../data/loaders/instagram/README_instagram.md) —
  the Instagram scraper
- [`../training/disentangled/README.md`](../training/disentangled/README.md) — the
  disentangled representation learning module (losses, usage, architecture)

---

## Map of the code

### Data and embeddings

| Path | Contents |
|---|---|
| `data/` | Dataset loaders (`loaders/`), preprocessing, embedding generators, storage adapters. See its README. |
| `data/exploratory-datasets/` | Pickled unified/audio/video datasets, with and without perturbations |
| `data/samples/` | A small sample embedding `.npz`, kept as a fixture |
| `embeddings/` | Precomputed embedding matrices (`.npy`), organized `audio/`, `video/`, `forensic/`, `audio_forensic/` by model |
| `labels/` | Label `.pkl` files matching the embedding matrices, same folder layout |
| `exports/` | Bulk embedding exports (gitignored) |

### Models and training

| Path | Contents |
|---|---|
| `models/model_factory.py` | Embedding-model factory and adapters (`ResidualMLP`, base model classes) |
| `models/configs/` | JSON model configs (`base_embeddings`, `orthogonal_model`, `direct_classifier`) |
| `models/time_series/` | Temporal transformer and 1D-CNN models over segment sequences |
| `models/pretrained/` | Vendored pretrained backbones — SENet, RIDNet |
| `pipeline/` | Encoder / base-model abstractions and sampling |
| `losses/` | Loss function definitions |
| `training/trainer.py` | Generic training loop |
| `training/disentangled/` | Disentangled representation learning: model, losses, metrics, and a unified train+eval `pipeline.py` |
| `transformer_experiments/` | Transformer and LSTM sequence classifiers over segment embeddings, plus their training variants (k-fold, stratified, cross-dataset, all-datasets) |
| `experiments/` | MLP classifiers, grid search, and the `fix1`/`fix1a` disentanglement repair experiments |
| `configs/` | YAML training configs (`train_audio`, `train_multimodal`, `train_forensic_only`). `config.yaml` holds credentials and is gitignored — copy `config_template.yaml` to create it. |
| `checkpoints/` | Saved model checkpoints (gitignored) |

### Evaluation

| Path | Contents |
|---|---|
| `evaluation/evaluator.py` | Shared evaluation: linear probe, kNN, ROC-AUC |
| `evaluation/embeddings/` | Linear-probe evaluation, correlation matrices, pairwise embedding-combination evaluation |
| `evaluation/downstream/` | Attack-type classification |

### Storage and infrastructure

| Path | Contents |
|---|---|
| `retriever/retriever.py` | Query embeddings + labels from Neon Postgres (pgvector) into numpy arrays |
| `db/` | SQLite/Neon embedding and model stores |
| `dropbox_utils/` | Dropbox client and token handling |
| `migrations/` | Database migrations |
| `utils/` | Config loading, embedding helpers, save helpers |
| `scripts/` | Standalone utilities: dataset zip loaders, Neon/DB setup and migration, batch embedding jobs, t-SNE and statistics, prediction inspection |

### Outputs

| Path | Contents |
|---|---|
| `results/` | All experiment outputs, grouped by theme — see [`../results/README.md`](../results/README.md) |
| `notebooks/` | Jupyter and Colab notebooks — see [`../notebooks/README.md`](../notebooks/README.md) |
| `logs/` | Job logs |

---

## Entry points

Most training modules are run as packages from the repository root, so imports like
`from utils.…` and `from training.…` resolve:

```bash
# Disentangled representation learning — train and evaluate in one run,
# sweeping conservative/moderate/aggressive hyperparameter configs
python -m training.disentangled.pipeline

# Single disentangled training run
python -m training.disentangled.main --hdf5-path <...> --encoder-name hubert

# Transformer sequence classifier
python -m transformer_experiments.train
python -m transformer_experiments.train_stratified
python -m transformer_experiments.train_all_datasets

# Configured experiments driven by models/configs/*.json
python experiments/run_experiment.py orthogonal_model
python experiments/main.py --mode single --config models/configs/orthogonal_model.json
```

**Run everything from the repository root.** Internal imports are absolute
(`from utils.…`, `from training.…`), so they only resolve when the root is on
`sys.path` — either by running from root, or by `pip install -e .` (see `setup.py`).

### Loose scripts at the repository root

These are standalone tools, deliberately left at root because they resolve paths
relative to it:

| File | Purpose |
|---|---|
| `main.py` | Orchestrates the AVDeepfake download → segment extraction → embedding generation pipeline |
| `inspect_hdf5.py` | Prints the structure and contents of an embeddings HDF5 file |
| `analyze_dataset_split.py` | Reports the real/fake and per-dataset composition of an HDF5 split |
| `test_model_detailed.py` | Detailed multi-dataset model testing — see [`guides/TEST_SCRIPT_USAGE.md`](guides/TEST_SCRIPT_USAGE.md) |
| `QUICK_START_TEST.sh` | Convenience wrapper around `test_model_detailed.py`; expects to be run from root |
| `test_dropbox_auth.py`, `test_dropbox_client.py` | Manual checks that Dropbox credentials and downloads work |

Despite the `test_` prefix, none of these are automated tests — there is no test suite
in this repository (`tests/` is an empty placeholder).

---

## Setup

```bash
pip install -r requirements.txt   # see also constraints.txt
pip install -e .                  # makes the packages importable from anywhere
cp configs/config_template.yaml configs/config.yaml   # then fill in credentials
```

Note that `.venv/` is **not** in `.gitignore` — avoid `git add -A` at the repository
root, or use `git add -A -- ':!.venv'`.
