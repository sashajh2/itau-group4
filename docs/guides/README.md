# Guides

How to run things. These are usage instructions, not design documents — for design see
[`../architecture/`](../architecture/).

| Document | Covers |
|---|---|
| [`PIPELINE_USAGE.md`](PIPELINE_USAGE.md) | The unified train+evaluate pipeline (`training/disentangled/pipeline.py`), including hyperparameter sweeps and what each of the shared `data_utils` / `train_utils` / `eval_utils` modules does |
| [`TEST_SCRIPT_USAGE.md`](TEST_SCRIPT_USAGE.md) | `test_model_detailed.py` at the repository root — testing a trained model separately on AVDeepfake1M, ShareVeo3, and Sora2, with per-dataset metrics and confusion matrices |
| [`Disentangled_Representation_Learning_Implementation_Guide.md`](Disentangled_Representation_Learning_Implementation_Guide.md) | Long-form implementation guide for the disentangled representation learning approach (thesis Section 3.3.2), treating each temporal segment as an independent sample |

## Before running anything

Run commands **from the repository root**. Internal imports are absolute
(`from utils.…`, `from training.…`), so they only resolve when the root is on
`sys.path` — either by running from root, or via `pip install -e .`.

Paths written in these guides are relative to the repository root and were correct when
written; if one no longer resolves, the file has likely moved rather than disappeared —
check [`../README.md`](../README.md) for the current layout.
