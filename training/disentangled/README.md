# Disentangled Representation Learning

Implementation of disentangled representation learning for deepfake detection as described in Section 3.3.2 of the thesis.

## Overview

This module implements a two-stage approach:
1. **Stage 1 (this module)**: Disentangle identity and authenticity representations
2. **Stage 2**: Temporal classification using disentangled embeddings

## Architecture

- **Dual Projection Heads**: Two 2-layer MLPs (f_auth and f_id) that project pretrained embeddings into separate subspaces
- **Input**: Pretrained embeddings (e.g., Hubert, OpenL3, SENET)
- **Output**: Two 128-dimensional embeddings (z^auth and z^id), both L2-normalized

## Loss Functions

1. **Variance Minimization** (Equation 3.6): Forces real samples to cluster tightly in z^auth space
2. **Prototypical Contrastive** (Equation 3.5): Clusters samples by content group (source_idx, segment_idx) in z^id space
3. **Orthogonality Constraint** (Equation 3.3): Enforces independence between z^id and z^auth

## Usage

### Basic Training

```bash
python -m training.disentangled.main \
    --hdf5-path exports/deepfake_embeddings_2.h5 \
    --encoder-name hubert \
    --batch-size 128 \
    --num-epochs 50 \
    --lr 1e-4
```

### Full Example with All Options

```bash
python -m training.disentangled.main \
    --hdf5-path exports/deepfake_embeddings_2.h5 \
    --encoder-name hubert \
    --input-dim 768 \
    --output-dim 128 \
    --batch-size 128 \
    --num-epochs 50 \
    --lr 1e-4 \
    --lambda-var 0.5 \
    --lambda-orth 0.1 \
    --temperature 0.1 \
    --num-workers 4 \
    --val-split 0.1 \
    --save-dir ./checkpoints/disentangled \
    --device cuda
```

### Arguments

**Data:**
- `--hdf5-path`: Path to HDF5 file with embeddings (required)
- `--encoder-name`: Encoder to use (`hubert`, `openl3`, or `senet`, default: `hubert`)

**Model:**
- `--input-dim`: Input embedding dimension (default: 768 for hubert)
- `--output-dim`: Output projection dimension (default: 128)

**Training:**
- `--batch-size`: Batch size (default: 128, recommend 128-256)
- `--num-epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--num-workers`: Number of data loading workers (default: 4)
- `--val-split`: Validation split ratio (default: 0.1)

**Loss Hyperparameters:**
- `--lambda-var`: Weight for variance loss (default: 0.5)
- `--lambda-orth`: Weight for orthogonality loss (default: 0.1)
- `--temperature`: Temperature for prototypical loss (default: 0.1)

**Other:**
- `--device`: Device to use (`cuda` or `cpu`, default: auto-detect)
- `--save-dir`: Directory to save checkpoints (default: `./checkpoints/disentangled`)

## Output

The training script will:
1. Load the HDF5 dataset and create train/val splits
2. Train the model for the specified number of epochs
3. Save checkpoints:
   - `best_model.pt`: Best model based on validation loss
   - `latest_model.pt`: Latest model after each epoch

Checkpoints contain:
- Model state dict
- Optimizer state dict
- Training/validation losses
- Hyperparameters

## Dataset Structure

The HDF5 file should have the following structure:
```
/videos/{video_id}/
    embeddings/
        hubert/  # [num_augs, num_segs, emb_dim]
        openl3/
        senet/
    labels/
        audio/   # [num_augs, num_segs]
        video/   # [num_augs, num_segs]
    augmentation_info/
        (attrs) source_idx  # Index of source video
```

## Next Steps

After training, you can:
1. Extract z^auth and z^id embeddings for all samples
2. Use z^auth for temporal classification (Stage 2)
3. Evaluate disentanglement quality using metrics from Section 3.4.3

## Files

- `dataset.py`: HDF5 dataset class and collate function
- `losses.py`: Three loss functions + total loss computation
- `model.py`: DisentangledProjector architecture
- `train.py`: Training and validation loops
- `main.py`: Main training script

