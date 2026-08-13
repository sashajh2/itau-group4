# Pipeline Usage Guide

## Overview

The unified pipeline (`training/disentangled/pipeline.py`) trains and evaluates models in one run, with support for hyperparameter sweeps.

## Architecture

- **`data_utils.py`**: Shared data loading (extracted from `evaluate_metrics.py`)
- **`train_utils.py`**: Reusable training function (works with pre-loaded data)
- **`eval_utils.py`**: Reusable evaluation functions:
  - `evaluate_single_dataset()`: Matches `evaluate_metrics.py` (includes local content-group metrics)
  - `evaluate_cross_dataset()`: Matches `evaluate_cross_dataset.py` (NO local content-group metrics)
- **`pipeline.py`**: Main orchestrator

## Usage

### Single Config Run

```bash
python3 -m training.disentangled.pipeline \
    --train-hdf5 exports/deepfake_embeddings_2.h5 \
    --test-hdf5 exports/sora2_embeddings.h5 \
    --encoder-name hubert \
    --output-dir results/pipeline_run_001 \
    --config conservative
```

### Hyperparameter Sweep (All 3 Configs)

```bash
python3 -m training.disentangled.pipeline \
    --train-hdf5 exports/deepfake_embeddings_2.h5 \
    --test-hdf5 exports/sora2_embeddings.h5 \
    --encoder-name hubert \
    --output-dir results/pipeline_sweep_001 \
    --run-hyperparameter-sweep
```

### With Custom Training Settings

```bash
python3 -m training.disentangled.pipeline \
    --train-hdf5 exports/deepfake_embeddings_2.h5 \
    --test-hdf5 exports/sora2_embeddings.h5 \
    --encoder-name hubert \
    --output-dir results/pipeline_run_002 \
    --num-epochs 100 \
    --batch-size 256 \
    --lr 5e-5 \
    --run-hyperparameter-sweep
```

## Hyperparameter Configs

The pipeline includes 3 pre-defined configs focused on variance regularization:

### Conservative
- `min_variance`: 0.1
- `variance_reg_weight`: 1.0

### Moderate
- `min_variance`: 0.2
- `variance_reg_weight`: 2.0

### Aggressive
- `min_variance`: 0.5
- `variance_reg_weight`: 5.0

**Note**: No separation loss (`lambda_sep=0`) - anomaly detection approach (real-only training)

## Output Structure

```
results/disentangled/pipeline_sweep_001/
├── all_results.json              # Combined results for all configs
├── conservative/
│   ├── best_model.pt            # Trained model
│   ├── metrics_history.json     # Training history
│   └── results.json             # Evaluation results
├── moderate/
│   ├── best_model.pt
│   ├── metrics_history.json
│   └── results.json
└── aggressive/
    ├── best_model.pt
    ├── metrics_history.json
    └── results.json
```

## Results JSON Structure

### Individual Config Results (`{config}/results.json`)

```json
{
  "config": {
    "min_variance": 0.1,
    "variance_reg_weight": 1.0,
    ...
  },
  "checkpoint_path": "...",
  "train_metrics": {
    "input_metrics": {
      "ami": 0.111,
      "ari": 0.066,
      "wasserstein_distance": 0.533,
      "intra_group_variance_real_mean": 13.78,
      ...
    },
    "model_metrics": {
      "ami": 0.105,
      "ari": 0.038,
      "wasserstein_distance": 0.002,
      "intra_group_variance_real_mean": 0.02,
      ...
    }
  },
  "cross_dataset_metrics": {
    "ami_vs_train_real_input": 0.016,
    "ami_vs_train_real_z_auth": 0.017,
    "wasserstein_distance_cross_dataset_input": 0.219,
    "wasserstein_distance_cross_dataset_z_auth": 0.001,
    ...
  }
}
```

## Key Differences from Original Scripts

### `evaluate_single_dataset()` vs `evaluate_metrics.py`
- ✅ **Same metrics**: Includes clustering, distribution, separation, AND local content-group metrics
- ✅ **Same structure**: Returns `{'input_metrics': ..., 'model_metrics': ...}`
- ✅ **Accepts pre-loaded data**: No need to load from HDF5 again

### `evaluate_cross_dataset()` vs `evaluate_cross_dataset.py`
- ✅ **Same metrics**: Includes clustering, distribution, separation
- ❌ **NO local content-group metrics**: As per your requirement
- ✅ **Same metric names**: Matches existing script exactly
- ✅ **Accepts pre-loaded data**: No need to load from HDF5 again

## Benefits

1. **Efficient**: Load datasets once, reuse for all configs
2. **No duplication**: Shared utilities across all scripts
3. **Easy comparison**: All configs in one place
4. **Modular**: Components can be used independently
5. **Backward compatible**: Existing scripts still work

## Next Steps

1. Run pipeline with hyperparameter sweep
2. Compare results across configs
3. Adjust hyperparameters if still collapsing
4. Use best config for final model

