# Hybrid Anomaly Detection for Deepfake Detection

## Goal
Build a temporal encoder with multiple objectives (reconstruction + contrastive) to learn the "real manifold", enabling generalization to unseen fake types (Sora, future methods).

## Architecture

```
[Audio + Video Sequences] → [Temporal Encoder] → [Shared Representation]
                                                         ↓
                          ┌──────────────────────────────┼──────────────────────────────┐
                          ↓                              ↓                              ↓
                 [Reconstruction Head]          [Contrastive Head]           [Classification Head]
                   (Decoder → MSE)              (Projector → InfoNCE)           (MLP → BCE)
                          ↓                              ↓                              ↓
                      L_recon                       L_contrast                      L_classify
```

## Training Strategy: Two-Stage (Recommended)

### Stage 1: Self-Supervised Pre-training (Real Data Only)
- **Data**: Real samples from AVDeepfake1M only (~221k samples)
- **Objectives**: `L_total = L_recon + L_contrast`
- **Duration**: 30-50 epochs
- **Encoder**: Fully trainable

### Stage 2: Classifier Fine-tuning
- **Data**: All data (real + fake)
- **Objectives**: `L_total = 0.5*L_recon + 0.5*L_contrast + 1.0*L_classify`
- **Duration**: 10-20 epochs
- **Encoder**: Frozen OR fine-tune with 10x smaller LR

## Loss Functions

| Loss | Formula | Purpose |
|------|---------|---------|
| Reconstruction | `MSE(x_recon, x_original)` | Learn to reconstruct real patterns |
| Contrastive | `InfoNCE(z_i, z_j)` where i,j from same video | Same-video segments cluster together |
| Classification | `BCE(logits, labels)` with pos_weight | Light supervision with fake data |

## Data Split

| Split | AVDeepfake1M | ShareVeo3 |
|-------|--------------|-----------|
| Train | 70% (real+fake) | Exclude |
| Val | 15% | Exclude |
| Test (ID) | 15% | Exclude |
| Test (OOD) | - | 100% (held out) |

## Files to Create

```
training/anomaly_detection/
├── model.py          # HybridAnomalyModel (extends time_series_model.py components)
├── heads.py          # ReconstructionHead, ContrastiveHead, ClassificationHead
├── losses.py         # reconstruction_loss, contrastive_loss, HybridLossNormalizer
├── train.py          # train_stage1, train_stage2
└── evaluate.py       # Anomaly score computation, AUROC, OOD evaluation
```

## Key Implementation Details

### ReconstructionHead
- Transformer decoder with cross-attention to latent
- Outputs both audio and video reconstructions
- MSE loss in embedding space (not raw audio/video)

### ContrastiveHead
- MLP projector to 128-dim space
- InfoNCE loss with temperature τ=0.07
- Positives: segments from same video (within temporal window)

### Anomaly Score (Inference)
```python
anomaly_score = reconstruction_error + alpha * (-classification_logit)
```
Higher score = more likely fake

## Evaluation Metrics
1. **AUROC** (primary) - on ID test set
2. **OOD AUROC** - real from ID vs fake from ShareVeo3
3. **FPR@95%TPR** - false positive rate at 95% detection
4. **Reconstruction error distribution** - real vs fake

## Files to Modify/Reuse
- `time_series_model.py` - Reuse `ModalityEncoder`, `PatchTokenizer`
- `training/disentangled/losses.py` - Follow `EqualWeightNormalizer` pattern
- `losses/loss.py` - Adapt `SupConLoss` for temporal contrastive

## Verification
1. Run Stage 1 training on real-only data, verify reconstruction loss decreases
2. Verify contrastive loss learns meaningful clusters (visualize with PCA)
3. Run Stage 2, verify classification metrics improve
4. Evaluate on held-out ShareVeo3 to test OOD generalization
5. Compare reconstruction error distribution: real vs fake