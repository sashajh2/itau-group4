# Detailed Model Testing Script - Usage Guide

## Overview

`test_model_detailed.py` provides comprehensive testing with detailed statistics for multiple datasets. It tests your trained model on AVDeepfake1M, ShareVeo3, and optionally Sora2 datasets separately, providing per-dataset metrics, confusion matrices, and detailed analysis.

## Features

- ✅ Test on multiple datasets (avdeepfake1m, shareveo3, sora2) separately or all at once
- ✅ Comprehensive metrics: Accuracy, Precision, Recall, F1-Score, AUROC
- ✅ Confusion matrix breakdown
- ✅ Per-batch statistics (saved for analysis)
- ✅ Per-sample predictions (optional)
- ✅ Probability distribution analysis
- ✅ Label vs prediction distribution
- ✅ Weighted averages across datasets
- ✅ JSON output for all results

## Quick Start

### Basic Usage (Test on AVDeepfake1M and ShareVeo3)

```bash
python test_model_detailed.py \
    --hdf5_path /path/to/deepfake_embeddings_2.h5 \
    --checkpoint_path ./checkpoints/best_model.pt
```

### Test on Specific Datasets

```bash
# Test only on AVDeepfake1M
python test_model_detailed.py \
    --hdf5_path /path/to/deepfake_embeddings_2.h5 \
    --checkpoint_path ./checkpoints/best_model.pt \
    --datasets avdeepfake1m

# Test on all three datasets
python test_model_detailed.py \
    --hdf5_path /path/to/deepfake_embeddings_2.h5 \
    --checkpoint_path ./checkpoints/best_model.pt \
    --datasets avdeepfake1m shareveo3 sora2

# Or use 'all' keyword
python test_model_detailed.py \
    --hdf5_path /path/to/deepfake_embeddings_2.h5 \
    --checkpoint_path ./checkpoints/best_model.pt \
    --datasets all
```

### Save Per-Sample Predictions

```bash
python test_model_detailed.py \
    --hdf5_path /path/to/deepfake_embeddings_2.h5 \
    --checkpoint_path ./checkpoints/best_model.pt \
    --save_predictions
```

### Custom Output Directory

```bash
python test_model_detailed.py \
    --hdf5_path /path/to/deepfake_embeddings_2.h5 \
    --checkpoint_path ./checkpoints/best_model.pt \
    --output_dir ./my_test_results
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hdf5_path` | str | **Required** | Path to HDF5 file containing test data |
| `--checkpoint_path` | str | **Required** | Path to model checkpoint (.pt file) |
| `--datasets` | list | `['avdeepfake1m', 'shareveo3']` | Datasets to test on: `avdeepfake1m`, `shareveo3`, `sora2`, or `all` |
| `--audio_embedding` | str | `openl3` | Audio embedding type: `openl3` or `hubert` |
| `--video_embedding` | str | `senet` | Video embedding type |
| `--use_audio_labels` | flag | True | Use audio labels (default: True) |
| `--batch_size` | int | 16 | Batch size for testing |
| `--output_dir` | str | `./test_results` | Directory to save results |
| `--save_predictions` | flag | False | Save per-sample predictions (first 1000 samples) |
| `--save_batch_stats` | flag | True | Save per-batch statistics |

## Output Files

The script creates JSON files in the output directory:

1. **Per-dataset results**: `test_results_{dataset_name}_{timestamp}.json`
   - Contains all metrics, confusion matrix, batch statistics for one dataset

2. **Combined results**: `test_results_combined_{timestamp}.json`
   - Contains results from all datasets plus weighted averages

## Output Format

### Console Output

The script prints:
- Dataset info (number of samples, segments)
- Overall metrics (Loss, Accuracy, Precision, Recall, F1, AUROC)
- Confusion matrix
- Label and prediction distributions
- Probability statistics
- Batch statistics summary
- Summary table across all datasets

### JSON Output

Each result file contains:

```json
{
  "dataset_name": "avdeepfake1m",
  "num_samples": 1000,
  "num_segments": 50000,
  "metrics": {
    "loss": 0.3119,
    "accuracy": 0.9646,
    "precision": 0.9821,
    "recall": 0.9472,
    "f1_score": 0.9643,
    "auroc": 0.9955
  },
  "confusion_matrix": {
    "true_negative": 25000,
    "false_positive": 500,
    "false_negative": 1300,
    "true_positive": 23200
  },
  "classification_report": { ... },
  "label_distribution": { "0": 25500, "1": 24500 },
  "prediction_distribution": { "0": 25300, "1": 24700 },
  "probability_statistics": {
    "mean": 0.5123,
    "std": 0.2345,
    "min": 0.0012,
    "max": 0.9987,
    "median": 0.5023,
    "q25": 0.3234,
    "q75": 0.7234
  },
  "batch_statistics": [ ... ]
}
```

## Example Output

```
======================================================================
Testing on AVDEEPFAKE1M
======================================================================
Loaded 5000 samples
Testing on avdeepfake1m: 100%|████████████| 313/313 [02:15<00:00,  2.31it/s]

======================================================================
TEST RESULTS: AVDEEPFAKE1M
======================================================================

Dataset Info:
  Number of samples: 5,000
  Number of segments: 250,000

Overall Metrics:
  Loss:           0.3119
  Accuracy:       0.9646 (96.46%)
  Precision:      0.9821
  Recall:         0.9472
  F1 Score:       0.9643
  AUROC:          0.9955

Confusion Matrix:
                  Predicted
                Fake    Real
  Actual Fake   25000      500
         Real    1300   23200

Label Distribution:
  Fake: 25,500 (10.20%)
  Real: 224,500 (89.80%)

...
```

## Use Cases

### 1. Compare Performance Across Datasets

```bash
python test_model_detailed.py \
    --hdf5_path embeddings.h5 \
    --checkpoint_path best_model.pt \
    --datasets all
```

This will show how your model performs on each dataset separately.

### 2. Detailed Analysis with Batch Statistics

The script automatically saves per-batch statistics. Use this to:
- Identify problematic batches
- Check for batch-level variations
- Analyze model consistency

### 3. Save Predictions for Further Analysis

```bash
python test_model_detailed.py \
    --hdf5_path embeddings.h5 \
    --checkpoint_path best_model.pt \
    --save_predictions
```

This saves per-sample predictions (first 1000) for detailed error analysis.

## Notes

1. **AUROC = 0.0**: If you see AUROC = 0.0, it likely means the dataset only contains one class (e.g., all fake in sora2). This is expected and not an error.

2. **Batch Statistics**: Per-batch statistics help identify if there are specific batches where the model struggles.

3. **Weighted Averages**: When testing on multiple datasets, the script computes weighted averages based on the number of segments in each dataset.

4. **GPU Usage**: The script automatically uses GPU if available. Make sure CUDA is properly configured.

## Troubleshooting

### Import Error

Make sure `time_series_model.py` is in the same directory or in your Python path:

```bash
export PYTHONPATH=/path/to/project:$PYTHONPATH
python test_model_detailed.py ...
```

### Dataset Not Found

If you get "No samples found for dataset", check:
- Dataset name spelling (must match exactly: `avdeepfake1m`, `shareveo3`, `sora2`)
- Dataset attribute in HDF5 file matches the filter
- HDF5 file path is correct

### Out of Memory

If you run out of GPU memory:
- Reduce `--batch_size` (try 8 or 4)
- Close other processes using GPU

