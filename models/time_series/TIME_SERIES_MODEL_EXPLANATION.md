# Time Series Model Explanation: PatchTST-style Temporal Transformer

## Overview

This document explains the `time_series_model.py` implementation, which is a **PatchTST-style temporal Transformer** for multimodal deepfake detection. The model processes audio and video embeddings across time to detect deepfakes at the segment level.

## Problem Statement

Traditional deepfake detection models often treat each frame/segment independently. However, deepfakes often have temporal inconsistencies that are easier to detect when considering the temporal context. This model:

1. **Processes sequences** of audio and video embeddings (not just single frames)
2. **Uses temporal patches** to capture local temporal patterns
3. **Separately encodes** audio and video modalities with Transformers
4. **Fuses** the two modalities to make per-segment predictions

## Architecture Overview

```
Input: Audio embeddings (T, d_a) + Video embeddings (T, d_v)
  ↓
[Patch Tokenization] → Audio patches (N, P*d_a) + Video patches (N, P*d_v)
  ↓
[Modality Encoders] → Audio tokens (N, D) + Video tokens (N, D)
  ↓
[Fusion Head] → Patch logits (N,)
  ↓
[Upsampling] → Segment logits (T,)
```

Where:
- **T** = number of time segments
- **N** = number of patches (N < T due to overlapping patches)
- **d_a, d_v** = embedding dimensions for audio/video
- **D** = model dimension (hidden size)
- **P** = patch size (segments per patch)

---

## Component Breakdown

### 1. Configuration (`ModelConfig`)

**Purpose**: Centralized configuration for all hyperparameters.

**Key Parameters**:
- **Embedding dimensions**: `audio_emb_dim=512` (OpenL3), `video_emb_dim=2048` (SENet)
- **Patch parameters**: `patch_size=8`, `patch_stride=4` (50% overlap)
- **Model architecture**: `model_dim=256`, `num_heads=8`, `num_layers=4`
- **Training**: Learning rate, batch size, epochs, etc.

**Why**: Makes it easy to experiment with different configurations without changing code.

---

### 2. Dataset (`AVH5Dataset`)

**Purpose**: Loads audio and video embeddings from HDF5 files.

**Key Features**:
- Loads sequences of embeddings (not single frames)
- Each augmentation is treated as a separate sample
- Supports filtering by dataset (`avdeepfake1m` vs `shareveo3`)
- Can use either audio or video labels

**Data Structure**:
```python
{
    "audio_seq": Tensor(T, 512),    # Audio embeddings over time
    "video_seq": Tensor(T, 2048),   # Video embeddings over time
    "label_seq": Tensor(T),         # Binary labels (0=fake, 1=real) per segment
    "video_id": str,
    "aug_idx": int,
}
```

**Why**: 
- HDF5 is efficient for large-scale data
- Treating augmentations separately increases training data
- Binary labels (thresholded at 0.5) simplify the classification task

---

### 3. Patch Tokenizer (`PatchTokenizer`)

**Purpose**: Converts time-series sequences into overlapping patches.

**How it works**:
1. Uses `unfold` to create sliding windows over the time dimension
2. Each patch contains `patch_size` consecutive segments
3. Patches overlap by `patch_stride` (e.g., stride=4, size=8 → 50% overlap)
4. Can either flatten patches (`P*d`) or mean-pool them (`d`)

**Example**:
```
Input: (B, T=20, d=512)
Patch size: 8, Stride: 4

Patches:
- Patch 0: segments [0:8]
- Patch 1: segments [4:12]
- Patch 2: segments [8:16]
- Patch 3: segments [12:20]

Output: (B, N=4, P*d) or (B, N=4, d) if mean-pooling
```

**Why patches?**:
- **Efficiency**: Reduces sequence length from T to N (N < T)
- **Local patterns**: Captures short-term temporal dependencies
- **Overlap**: Ensures no information is lost at boundaries
- **Inspired by PatchTST**: Proven effective for time-series tasks

**Mean pooling vs flattening**:
- **Flattening**: `(B, N, P*d)` - preserves all information, but increases dimensionality
- **Mean pooling**: `(B, N, d)` - reduces dimensionality, but loses fine-grained temporal info
- **Current choice**: Mean pooling when `patch_size > 1` to keep dimensions manageable

**Segment-to-patches mapping**:
- Tracks which patches cover each segment
- Used later for upsampling patch predictions back to segment level

---

### 4. Modality Encoder (`ModalityEncoder`)

**Purpose**: Encodes patches for a single modality (audio or video) using a Transformer.

**Components**:

1. **Patch Projection**: `Linear(patch_dim → model_dim)`
   - Projects patches to a common dimension D (e.g., 256)

2. **Positional Encoding**: Learnable positional embeddings
   - Max length: 1000 patches (configurable)
   - Adds temporal position information

3. **Transformer Encoder**: Standard PyTorch `TransformerEncoder`
   - Multi-head self-attention
   - Feed-forward network
   - Layer normalization and dropout

**Architecture**:
```
Input patches: (B, N, patch_dim)
  ↓ [Patch Projection]
(B, N, model_dim)
  ↓ [+ Positional Encoding]
(B, N, model_dim)
  ↓ [Transformer Encoder (num_layers)]
(B, N, model_dim)  ← Contextualized tokens
```

**Why separate encoders?**:
- Audio and video have different characteristics
- Allows each modality to learn modality-specific temporal patterns
- More flexible than a single shared encoder

**Why Transformer?**:
- **Self-attention**: Captures long-range dependencies
- **Parallel processing**: Efficient for sequences
- **Proven**: State-of-the-art for sequence modeling

---

### 5. Fusion Head (`FusionHead`)

**Purpose**: Combines audio and video tokens to produce per-patch predictions.

**Architecture**:
```
Audio tokens: (B, N, D)
Video tokens: (B, N, D)
  ↓ [Concatenate]
(B, N, 2*D)
  ↓ [Fusion MLP]
(B, N, 1)
  ↓ [Squeeze]
(B, N)  ← Patch logits
```

**Fusion MLP**:
- Input: `2*model_dim` (concatenated audio + video)
- Hidden layers: `hidden_dim → hidden_dim//2 → 1`
- Activation: GELU
- Dropout for regularization

**Why concatenation + MLP?**:
- Simple and effective
- Allows the model to learn how to combine modalities
- MLP can learn complex interactions between audio and video features

**Alternative considered**: Cross-attention between modalities (more complex, not implemented)

---

### 6. Upsampling (`upsample_patch_logits`)

**Purpose**: Converts patch-level predictions back to segment-level predictions.

**Problem**: Patches overlap, so each segment may be covered by multiple patches.

**Solution**: For each segment, aggregate logits from all covering patches.

**Methods**:
- **Average**: Mean of all covering patch logits (current default)
- **Max**: Maximum of all covering patch logits

**Example**:
```
Segment 5 is covered by:
- Patch 0 (segments 0-7)
- Patch 1 (segments 4-11)

Segment 5 logit = average(Patch 0 logit, Patch 1 logit)
```

**Why average?**:
- Smooth predictions
- Reduces noise from individual patches
- Works well in practice

---

### 7. Full Model (`AVTemporalModel`)

**Purpose**: Combines all components into a complete model.

**Forward pass**:
```python
def forward(audio_seq, video_seq):
    # 1. Tokenize into patches
    audio_patches, patch_info = audio_tokenizer(audio_seq)
    video_patches, _ = video_tokenizer(video_seq)
    
    # 2. Encode patches
    audio_tokens = audio_encoder(audio_patches)
    video_tokens = video_encoder(video_patches)
    
    # 3. Fuse and get patch logits
    patch_logits = fusion_head(audio_tokens, video_tokens)
    
    # 4. Upsample to segment level
    segment_logits = upsample_patch_logits(patch_logits, patch_info)
    
    return segment_logits
```

**Output**: `(B, T)` logits - one per segment (real/fake prediction)

---

## Training Details

### Loss Function

**BCEWithLogitsLoss**: Binary cross-entropy with logits
- Input: Segment logits `(B, T)`
- Target: Binary labels `(B, T)` (0=fake, 1=real)
- Handles variable-length sequences via masking

**Masking**:
- Sequences are padded to the same length in batches
- `valid_length` tensor tracks actual lengths
- Padding is masked out in loss computation

### Metrics

1. **AUROC**: Area under ROC curve (primary metric)
2. **Accuracy**: Binary classification accuracy
3. **Loss**: Training/validation loss

### Optimizer

**AdamW** with:
- Learning rate: `1e-4`
- Weight decay: `1e-5` (L2 regularization)

### Data Handling

**Collate function**:
- Pads sequences to max length in batch
- Tracks `valid_length` for masking
- Returns batched tensors

**Dataset splitting**:

**Important Dataset Characteristics**:
- **ShareVeo3**: Contains **ALL deepfake videos** (100% fake)
- **AVDeepfake1M**: Contains **both real and fake videos** (mixed labels)

This distinction is crucial for understanding the split strategy and potential biases.

The dataset is divided in the following way:

1. **Load two separate datasets**:
   - `AVDeepfake1M` dataset (filtered by `filter_dataset="avdeepfake1m"`)
     - Contains both real and fake video segments
   - `ShareVeo3` dataset (filtered by `filter_dataset="shareveo3"`)
     - Contains only fake video segments

2. **Combine datasets**:
   - Uses `torch.utils.data.ConcatDataset` to combine both datasets
   - Creates a single unified dataset from both sources

3. **Random 80/20 split**:
   - `train_size = int(0.8 * len(combined_dataset))`
   - `val_size = len(combined_dataset) - train_size`
   - Uses `torch.utils.data.random_split()` for the split
   - **Important**: This is a **random split**, meaning samples from both AVDeepfake and ShareVeo3 can appear in both train and validation sets

4. **Sample granularity**:
   - Each augmentation is treated as a separate sample
   - If a video has 5 augmentations, it contributes 5 samples to the dataset

**Example**:
```
AVDeepfake1M: 1000 samples (mix of real/fake)
ShareVeo3: 500 samples (all fake)
Combined: 1500 samples
  ↓ (random split)
Train: 1200 samples (80%)
Val: 300 samples (20%)
```

**Implications of Current Split Strategy**:

⚠️ **Potential Issues**:
1. **Class imbalance**: Since ShareVeo3 is all fake, randomly splitting might create uneven class distributions in train/val
2. **Validation representativeness**: The validation set might not accurately reflect the true real/fake distribution if it gets a disproportionate number of ShareVeo3 samples
3. **Dataset bias**: The model might learn dataset-specific features rather than general deepfake detection features

**Better Alternatives** (not currently implemented):

1. **Stratified split by class**:
   - Split to maintain similar real/fake ratios in train and val
   - Ensures validation set is representative

2. **Split datasets separately, then combine**:
   - Split AVDeepfake 80/20 (maintaining class balance)
   - Split ShareVeo3 80/20
   - Combine train sets and combine val sets
   - Ensures both datasets are represented in both splits

3. **Use ShareVeo3 as test-only**:
   - Train/val only on AVDeepfake (which has both classes)
   - Use ShareVeo3 as a separate test set to evaluate generalization

**Current Note**: The random split is **not stratified** by dataset source or class label, so the train/val distribution may not be balanced. This could affect model evaluation and generalization.

---

## Design Decisions & Rationale

### 1. Why PatchTST-style?

- **Efficiency**: Reduces sequence length, making Transformers more efficient
- **Local patterns**: Captures short-term temporal dependencies
- **Proven**: PatchTST is state-of-the-art for time-series forecasting

### 2. Why Separate Modality Encoders?

- **Modality-specific patterns**: Audio and video have different temporal characteristics
- **Flexibility**: Can use different architectures/configurations per modality
- **Interpretability**: Easier to understand what each modality contributes

### 3. Why Mean Pooling for Patches?

- **Dimensionality**: Prevents explosion when `patch_size * emb_dim` is large
- **Simplicity**: Keeps model dimension constant regardless of patch size
- **Trade-off**: Loses fine-grained temporal info, but gains efficiency

### 4. Why Overlapping Patches?

- **No information loss**: Every segment is covered
- **Smoother predictions**: Reduces boundary effects
- **Redundancy**: Helps model learn robust features

### 5. Why Per-Segment Predictions?

- **Fine-grained**: Can detect which segments are fake
- **Temporal localization**: Identifies when deepfakes occur in video
- **More informative**: Better than video-level predictions

---

## Technical Details

### Memory Efficiency

- **Mean pooling**: Reduces memory usage for large patch sizes
- **Batch processing**: Efficient GPU utilization
- **Gradient checkpointing**: Not implemented (could be added for very large models)

### Sequence Length Handling

- **Variable lengths**: Handled via padding and masking
- **Max length**: No hard limit, but very long sequences may cause memory issues
- **Positional encoding**: Supports up to 1000 patches (configurable)

### Model Size

Typical parameter breakdown:
- **Audio Encoder**: ~1-2M parameters
- **Video Encoder**: ~1-2M parameters
- **Fusion Head**: ~100K parameters
- **Total**: ~3-5M parameters (depends on config)

### Training Considerations

1. **Label thresholding**: Soft labels → binary (threshold 0.5)
2. **Class imbalance**: May need weighting if severe
3. **Augmentation**: Each augmentation is a separate sample
4. **Validation**: Saves best model based on validation AUROC

---

## Usage Example

```python
# Create config
config = ModelConfig(
    audio_emb_dim=512,
    video_emb_dim=2048,
    patch_size=8,
    patch_stride=4,
    model_dim=256,
    num_heads=8,
    num_layers=4,
)

# Create model
model = AVTemporalModel(config)

# Forward pass
audio_seq = torch.randn(B, T, 512)  # (batch, time, audio_dim)
video_seq = torch.randn(B, T, 2048)  # (batch, time, video_dim)
segment_logits, _ = model(audio_seq, video_seq)  # (B, T)
```

---

## Future Improvements

1. **Cross-modal attention**: Add attention between audio and video tokens
2. **Hierarchical patches**: Multi-scale patches (short + long-term)
3. **Temporal pooling**: Global pooling for video-level predictions
4. **Pretraining**: Pretrain on large-scale video datasets
5. **Ensemble**: Combine multiple models for better performance

---

## Summary

This model implements a **temporal Transformer** that:
1. **Processes sequences** of audio and video embeddings
2. **Uses patches** to capture local temporal patterns efficiently
3. **Encodes modalities separately** with Transformers
4. **Fuses** audio and video to make per-segment predictions
5. **Handles variable-length sequences** via padding and masking

The architecture is inspired by PatchTST (time-series forecasting) but adapted for multimodal deepfake detection. It's designed to be efficient, flexible, and effective for detecting temporal inconsistencies in deepfakes.

