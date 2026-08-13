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
`pipeline_aggressive`, `train_only`, `fix1_variants`, `fix1a_repulsion`. Each sweep
directory contains `conservative/`, `moderate/`, `aggressive/` subdirectories and an
`all_results.json`.

**`classifiers/`** — `concat_mlp`, `concat_mlp_class_weights`, `concat_mlp_oversampling`,
`concat_mlp_oversampling_4fold`, `mlp_classifier`.

## Known stale paths after the regrouping

Nothing here is imported, so no code was broken at import time. Path references inside
`docs/` were updated to the new locations. But a handful of **scripts** still hardcode
the old flat `results/` paths. Each is a one-line fix, left undone deliberately: the
reorganization did not change any Python source, so no runtime behavior shifted
underneath you.

| File | Line | Points at | Should now be |
|---|---|---|---|
| `transformer_experiments/eval_200.py` | 34 | `results/transformer/kfold_5fold_hubert_fold5_….pt` | `results/sequence_models/transformer/…` |
| `scripts/inspect_predictions.py` | 30–34 | five `results/transformer/kfold_5fold_hubert_fold*.pt` | `results/sequence_models/transformer/…` |
| `scripts/local_global_stats.py` | 3 | absolute path ending `…/results/baseline` | `…/results/embedding_analysis/baseline` |

All three reference gitignored `.pt` checkpoints that are not in the repository anyway,
so they already required a local re-run to be useful.

Scripts that *write* into `results/` — `experiments/fix1_variants_experiment.py`,
`experiments/fix1a_repulsion_experiment.py`, `experiments/concat_mlp_experiment.py`,
`experiments/train_mlp_classifier.py`, `scripts/run_baseline_experiments.py` — still
have their old flat output paths. They will not fail; they will simply recreate a
top-level directory such as `results/fix1_variants/` on the next run. Update their
output-directory constants when convenient.
