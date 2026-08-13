# Parameter Breakdown for AVTemporalModel

## Current Configuration (7.5M parameters)

### Main Contributors:

1. **Audio Encoder (~3.5M)**:
   - Patch projection: 512 × 256 = 131K
   - Positional encoding: 1000 × 256 = 256K
   - Transformer (4 layers):
     - Self-attention: 4 × 256² = 262K per layer
     - FFN: 2 × (256 × 1024) = 524K per layer
     - Total per layer: ~786K
     - 4 layers: ~3.1M

2. **Video Encoder (~3.9M)**:
   - Patch projection: 2048 × 256 = 524K (LARGE!)
   - Positional encoding: 1000 × 256 = 256K
   - Transformer: same as audio = ~3.1M

3. **Fusion Head (~400K)**:
   - MLP: 512 → 256 → 1

## Ways to Reduce Parameters:

### Option 1: Reduce Model Dimension (Easiest)
- Change `model_dim=256` → `model_dim=128`
- Reduces parameters by ~4x
- **New total: ~2M parameters**

### Option 2: Fewer Layers
- Change `num_layers=4` → `num_layers=2`
- Reduces parameters by ~50%
- **New total: ~4M parameters**

### Option 3: Smaller FFN
- Change `dim_feedforward=1024` → `dim_feedforward=512`
- Reduces parameters by ~25%
- **New total: ~5.5M parameters**

### Option 4: Reduce Video Embedding Dimension First
- Add a projection layer: 2048 → 512 before patch tokenizer
- Then use smaller patch projection
- **New total: ~4M parameters**

### Option 5: Shared Encoder (Most Aggressive)
- Use single encoder for both modalities
- Add modality embedding to distinguish them
- **New total: ~4M parameters**

## Recommended: Combination Approach

```python
config = ModelConfig(
    model_dim=128,        # Reduced from 256
    num_layers=3,         # Reduced from 4
    dim_feedforward=512,  # Reduced from 1024
    # ... rest stays same
)
# Expected: ~2-3M parameters
```

