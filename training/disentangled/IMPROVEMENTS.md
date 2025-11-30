# Disentangled Learning Improvements

## Summary of Changes

This document describes the improvements made to address training instability and loss imbalance issues.

## 1. Content Group Filtering

**Problem**: Prototypical learning is unstable when batches contain content groups with only 1-2 samples.

**Solution**: Filter dataset to only include content groups with at least `min_samples_per_group` samples (default: 3).

**Files Modified**:
- `dataset.py`: Added `min_samples_per_group` parameter and filtering logic
- Added content group statistics logging

**Usage**:
```python
dataset = DisentanglementDataset(
    hdf5_path, 
    encoder_name='hubert',
    min_samples_per_group=3  # Only include groups with ≥3 samples
)
```

**Analysis Script**:
```bash
python -m training.disentangled.analyze_content_groups \
    --hdf5-path exports/deepfake_embeddings.h5 \
    --encoder-name hubert \
    --output content_group_analysis.csv
```

## 2. Regularized Variance Loss

**Problem**: Variance loss collapses to near-zero (5.9e-08), making `lambda_var` ineffective.

**Solution**: Add regularization term that penalizes variance below a minimum threshold.

**Implementation**:
```python
variance = ((z_auth_real - mu_real) ** 2).sum(dim=1).mean()
regularization = torch.clamp(min_variance - variance, min=0.0) ** 2
loss = variance + regularization_weight * regularization
```

**Parameters**:
- `min_variance` (default: 0.01): Minimum variance threshold
- `variance_reg_weight` (default: 0.1): Weight for regularization term

**Files Modified**:
- `losses.py`: Updated `variance_loss()` function

## 3. Adaptive Loss Scaling

**Problem**: Loss magnitudes are imbalanced (proto ~0.087, var ~5.9e-08, orth ~0.00028), so fixed weights don't work well.

**Solution**: Use exponential moving average to normalize losses before weighting.

**Implementation**:
- `AdaptiveLossScaler` class tracks running averages of each loss
- Normalizes losses by their running averages before applying weights
- Smoothing factor `alpha=0.99` for stability

**Files Modified**:
- `losses.py`: Added `AdaptiveLossScaler` class
- `train.py`: Integrated adaptive scaler into training loop
- `main.py`: Added `--use-adaptive-scaling` flag (enabled by default)

**Usage**:
```bash
# Enable (default)
python -m training.disentangled.main --hdf5-path ... --use-adaptive-scaling

# Disable
python -m training.disentangled.main --hdf5-path ... --no-adaptive-scaling
```

## 4. Enhanced Logging

**New Logging Features**:
- Content group size statistics (mean, median, min, max, percentiles)
- Scaled loss values (when adaptive scaling is enabled)
- Loss scaling factors

**Example Output**:
```
📊 Content Group Statistics (after filtering):
   Unique content groups: 45,231
   Mean samples per group: 4.2
   Median samples per group: 4.0
   Min samples per group: 3
   Max samples per group: 20
   Groups with ≥2 samples: 45,231
   Groups with ≥3 samples: 45,231
   Groups with ≥5 samples: 12,456
```

## New Command-Line Arguments

### Dataset Arguments
- `--min-samples-per-group`: Minimum samples per content group (default: 3)

### Loss Arguments
- `--min-variance`: Minimum variance threshold (default: 0.01)
- `--variance-reg-weight`: Weight for variance regularization (default: 0.1)
- `--use-adaptive-scaling`: Enable adaptive loss scaling (default: True)
- `--no-adaptive-scaling`: Disable adaptive loss scaling

## Recommended Workflow

### Step 1: Analyze Content Groups
```bash
python -m training.disentangled.analyze_content_groups \
    --hdf5-path exports/deepfake_embeddings.h5 \
    --encoder-name hubert \
    --output content_group_analysis.csv
```

Review the output to determine appropriate `min_samples_per_group` value.

### Step 2: Train with Improvements
```bash
python -m training.disentangled.main \
    --hdf5-path exports/deepfake_embeddings.h5 \
    --encoder-name hubert \
    --min-samples-per-group 3 \
    --min-variance 0.01 \
    --variance-reg-weight 0.1 \
    --use-adaptive-scaling \
    --lambda-var 0.5 \
    --lambda-orth 0.1 \
    --temperature 0.1 \
    --num-epochs 50
```

### Step 3: Monitor Training
Watch for:
- Variance loss staying above minimum threshold
- Scaled losses having similar magnitudes
- AMI/ARI metrics improving over time

## Expected Improvements

1. **Stable Prototypical Learning**: Filtering ensures each content group has multiple samples for stable prototype computation.

2. **Prevented Variance Collapse**: Regularization keeps variance above threshold, maintaining authenticity signal.

3. **Balanced Loss Contributions**: Adaptive scaling ensures all three losses contribute meaningfully to training.

4. **Better Metrics**: AMI and ARI should improve as the model learns better authenticity representations.

## Next Steps

1. Run content group analysis to determine optimal `min_samples_per_group`
2. Experiment with different `min_variance` and `variance_reg_weight` values
3. Tune `lambda_var` and `lambda_orth` with adaptive scaling enabled
4. Compare results with/without improvements to quantify gains

