# Results

Experiment outputs — plots, metrics JSON, CSVs, and model checkpoints. Previously 35 flat
sibling directories; now grouped by the kind of experiment that produced them.

**Model checkpoints (`*.pt`) are gitignored**, so a fresh clone has the plots and metrics
but not the weights. Anything that needs weights must be re-run.

| Group | What is in it | Produced by |
|---|---|---|
| [`embedding_analysis/`](embedding_analysis/) | Linear-probe scores, correlation matrices/heatmaps, and pairwise-combination analyses of the pretrained embeddings, per modality | `evaluation/embeddings/*`, `scripts/run_baseline_experiments.py`, `scripts/tsne_openl3.py` |
| [`sequence_models/`](sequence_models/) | Transformer and LSTM runs over segment sequences: confusion matrices, ROC curves, k-fold aggregates, checkpoints | `transformer_experiments/*` |
| [`disentangled/`](disentangled/) | Disentangled representation learning: hyperparameter sweeps and the `fix1`/`fix1a` collapse-repair experiments | `training/disentangled/pipeline.py`, `experiments/fix1*_experiment.py` |
| [`classifiers/`](classifiers/) | MLP classifiers on concatenated embeddings — class-weighting and oversampling variants, 4-fold runs | `experiments/concat_mlp_experiment.py`, `experiments/train_mlp_classifier.py` |
| [`scratch/`](scratch/) | **Ignorable.** Throwaway runs kept only so they are not confused with real results: `crash_test`, `timing_test5`, `timing_full` | — |

Two loose files stay at this level: `cross_dataset_metrics.json` and
`experiment_base_embeddings.json`.

## Contents by group

**`embedding_analysis/`** — `audio`, `audio_noise`, `video`, `forensic`,
`joint-video-forensic`, `baseline`, `tsne`, plus `content_group_analysis.csv` (the
real/fake breakdown per content group, output of
`training/disentangled/analyze_content_groups.py`).

**`sequence_models/`** — `transformer`, `transformer_all_datasets`,
`transformer_stratified`, `compare_pos_encoding`, `seq_len_sweep`, `eval_200`, `lstm`.

**`disentangled/`** — `pipeline_sweep_001`, `senet_pipeline_sweep_001`,
`pipeline_aggressive`, `train_only`, `fix1_variants`, `fix1a_repulsion`, `fix1_probe`
(logistic and linear-vs-nonlinear probes of the fix1 `z_auth` embeddings). Each sweep
directory contains `conservative/`, `moderate/`, `aggressive/` subdirectories and an
`all_results.json`.

**`classifiers/`** — `concat_mlp`, `concat_mlp_class_weights`, `concat_mlp_oversampling`,
`concat_mlp_oversampling_4fold`, `mlp_classifier`, `ensemble` (Hard-OR voting and a
random-forest meta-learner over the three per-modality logistic probes).

## Where scripts write

Every script that reads from or writes into `results/` was updated to the grouped
layout, so re-running anything lands in the right place rather than recreating a
top-level directory. Output directories are created with `exist_ok=True` and parents, so
the deeper paths need no setup.

| Writes into | Script |
|---|---|
| `sequence_models/transformer/` | `transformer_experiments/train.py`, `train_kfold.py`, `train_cross_dataset.py` |
| `sequence_models/transformer_stratified/` | `transformer_experiments/train_stratified.py` |
| `sequence_models/transformer_all_datasets/` | `transformer_experiments/train_all_datasets.py` |
| `sequence_models/transformer_single_embedding/` | `transformer_experiments/train_single_embedding.py` |
| `sequence_models/compare_pos_encoding/` | `transformer_experiments/compare_pos_encoding.py` |
| `sequence_models/seq_len_sweep/` | `transformer_experiments/test_kfold.py` |
| `sequence_models/eval_200/` | `transformer_experiments/eval_200.py` |
| `sequence_models/lstm/` | `transformer_experiments/lstm_train_hub.py`, `lstm_train_concat.py` |
| `classifiers/concat_mlp/` | `experiments/concat_mlp_experiment.py` |
| `classifiers/mlp_classifier/` | `experiments/train_mlp_classifier.py` |
| `disentangled/fix1_variants/` | `experiments/fix1_variants_experiment.py` |
| `disentangled/fix1a_repulsion/` | `experiments/fix1a_repulsion_experiment.py` |
| `embedding_analysis/baseline/` | `scripts/run_baseline_experiments.py` (`--output-dir` default) |
| `embedding_analysis/tsne/` | `scripts/tsne_openl3.py` |
| `classifiers/ensemble/` | `scripts/plot_ensemble_results.py` |
| `disentangled/fix1_probe/` | `scripts/probe_fix1_embeddings.py` |
| *(caller's choice)* | `training/disentangled/pipeline.py` — `--output-dir` is required |

Three scripts also *read* checkpoints from `sequence_models/transformer/`:
`transformer_experiments/eval_200.py`, `transformer_experiments/test_kfold.py`, and
`scripts/inspect_predictions.py`. They name specific timestamped `.pt` files which are
gitignored and therefore absent from a fresh clone — re-run the corresponding training
script to regenerate them, then update the filenames at the top of each script.

`scripts/local_global_stats.py` reads `embedding_analysis/baseline/` through a hardcoded
absolute path containing a user's home directory; change it to a relative path or your
own before running.
