# itau-group4 — Audiovisual Deepfake Detection

Research code for detecting deepfakes from audio and video, built around **pretrained
embeddings**: compute them once, cache them, and reuse them across every downstream
experiment.

```
raw video datasets  →  segments  →  pretrained embeddings  →  downstream models
 AVDeepfake1M          audio +      audio:  HuBERT, OpenL3,    linear probes
 ShareVeo3             video        wav2vec2, MFCC            MLP classifiers
 DFD, MovieGen         frames       video:  SENet, ArcFace,    temporal transformers
 SynVTA, Sora2                      FaceNet, MagFace, MARLIN     + LSTMs
                                    forensic: ResNet, RIDNet   disentangled repr.
```

Embeddings live as `.npy` matrices in `embeddings/` (with matching labels in `labels/`),
and are also stored in HDF5 files, a Neon Postgres pgvector database, and on Dropbox.
Most experiments start from these rather than from raw video.

## → [Full documentation index: `docs/README.md`](docs/README.md)

**Start there.** It maps every directory, names the entry points, and gives a reading
order. The short version:

| If you want to… | Read |
|---|---|
| Understand the system | [`docs/architecture/branch_documentation.md`](docs/architecture/branch_documentation.md) |
| Know what has been tried | [`docs/analysis/experiment_summary.md`](docs/analysis/experiment_summary.md) |
| See the deepest findings | [`docs/analysis/local_global_embedding_analysis.md`](docs/analysis/local_global_embedding_analysis.md) |
| Run the training pipeline | [`docs/guides/pipeline_usage.md`](docs/guides/pipeline_usage.md) |
| Find something to work on | [`docs/planning/ideas/`](docs/planning/ideas/) |

## Setup

```bash
pip install -r requirements.txt
pip install -e .                                       # makes packages importable anywhere
cp configs/config_template.yaml configs/config.yaml    # then fill in credentials
```

`configs/config.yaml` holds secrets and is gitignored.

## Running things

**Run from the repository root.** Intra-repo imports are absolute (`from utils.…`,
`from training.…`), so they resolve only when the root is on `sys.path` — either by
running from root, or via `pip install -e .`.

```bash
# Disentangled representation learning: trains and evaluates in one run,
# sweeping conservative / moderate / aggressive configs
python -m training.disentangled.pipeline

# Transformer classifier over segment sequences
python -m transformer_experiments.train
python -m transformer_experiments.train_stratified

# Experiments driven by models/configs/*.json
python experiments/run_experiment.py orthogonal_model
```

## Layout

```
data/                  dataset loaders, preprocessing, embedding generation
embeddings/ labels/    cached embedding matrices and their labels
models/                model factory, time-series models, pretrained backbones
training/              training loops; training/disentangled/ is the main module
transformer_experiments/  transformer and LSTM sequence classifiers
experiments/           MLP classifiers, grid search, fix1/fix1a repair experiments
evaluation/            linear probes, correlation matrices, downstream tasks
pipeline/ losses/ utils/   shared abstractions and helpers
retriever/ db/ dropbox_utils/   embedding storage and retrieval
scripts/               standalone CLI utilities
configs/               YAML training configs
docs/                  all documentation  ← start here
notebooks/             Jupyter and Colab notebooks
results/               experiment outputs, grouped by experiment type
```

Loose scripts at the root (`main.py`, `inspect_hdf5.py`, `analyze_dataset_split.py`,
`test_model_detailed.py`, `QUICK_START_TEST.sh`, `test_dropbox_*.py`) are standalone
tools that resolve paths relative to the root — they are described in
[`docs/README.md`](docs/README.md). Despite the `test_` prefix, none are automated
tests; there is no test suite in this repository.

## Conventions

- Documentation filenames are `lowercase_with_underscores.md`; `README.md` is the
  exception.
- Package-level `README.md` files stay next to the code they describe
  (`data/`, `data/loaders/`, `training/disentangled/`). Everything else is in `docs/`.
- Model checkpoints (`*.pt`), `exports/`, and `checkpoints/` are gitignored, so a fresh
  clone has plots and metrics but no weights.
