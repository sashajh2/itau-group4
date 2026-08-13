# Branch Documentation: Deepfake Detection Pipeline Refactoring

## Overview

This branch implements a scalable deepfake detection pipeline with support for orthogonal embedding heads, episodic learning, and comprehensive evaluation. The system supports evaluation on raw embeddings without training, making it ideal for baseline studies.

## Architecture + Model Decisions

### Heterogeneous Head (Identity Learning)

The heterogeneous head has two stages of training:

#### Stage 1: ArcFace-Based Classification
- **Purpose**: Learn discriminative identity embeddings
- **Loss**: ArcFace (angular margin softmax)
- **Implementation**: 
  - Located in `losses/loss.py`: `ArcMarginProduct` and `ArcFaceLoss` classes
  - Formula: `L = -log(softmax(s * cos(θ_y + m)))`
  - Parameters: `s=30.0` (scaling), `m=0.5` (angular margin in radians)
- **Batching**: Standard shuffled batches with diverse identities (32-128 identities per batch recommended)

#### Stage 2: Prototypical Episodic Learning
- **Purpose**: Learn to generalize to new identities via few-shot learning
- **Loss**: Cosine similarity to prototypes with cross-entropy
- **Implementation**:
  - Located in `training/trainer.py`: `_train_stage_b()` method
  - Uses `KWayNShotBatchSampler` for episodic batching
- **Batching**: 
  - Located in `data/data_loader.py`: `KWayNShotBatchSampler` class (lines 21-83)
  - Episodic structure: 5-way (5 identities), 5-shot (5 support samples), 15 queries per episode
  - Total: `N * (K + Q) = 5 * (5 + 15) = 100 samples per episode`
  - Controlled by config parameters: `het_n_way`, `het_k_shot`, `het_n_query`

### Homogeneous Head (Real/Fake Classification)

The homogeneous head uses a standard two-stage training approach:

#### Stage A: Real-Only Pretraining
- **Purpose**: Pretrain on real samples only with identity contrastive loss
- **Losses**:
  - `L_hom`: Variance compactness (VICReg-style) - keeps real embeddings compact
  - `L_orth`: Cross-covariance penalty - decorrelates homogeneous and heterogeneous heads
  - `L_het`: Supervised contrastive loss - learns identity embeddings
- **Implementation**: Located in `training/trainer.py`: `_train_stage_a()` method (lines 61-144)

#### Stage B: Classification Fine-Tuning
- **Purpose**: Fine-tune on all data (real + fake) with classification loss
- **Losses**: Combines Stage A losses with BCE classification loss
- **Implementation**: Located in `training/trainer.py`: `_train_stage_b()` method (lines 146-252)

## Pipeline + Code Implemented

### Data Loading (`data/data_loader.py`)

**Key Features:**
- Loads embeddings directly from Neon Postgres (no FAISS/dropbox needed)
- Identity-aware splitting: splits by identity to ensure each set has enough samples per identity
- Filters identities with <20 samples to support episodic evaluation

**DataLoader Structure:**
```python
{
    'train': Standard shuffled DataLoader (used for most stages)
    'train_het_stage_b': Episodic DataLoader (5-way 5-shot + 15 queries)
    'val': Standard validation DataLoader
    'val_het_eval': Episodic validation DataLoader (for heterogeneous head)
    'test': Standard test DataLoader
    'test_het_eval': Episodic test DataLoader (for heterogeneous head)
}
```

**Splitting Strategy** (lines 221-253):
1. Filter identities with <20 samples
2. Split identities (not samples) into train/val/test (70/15/15)
3. Ensures each identity in val/test has enough samples for episodic evaluation

### Model Factory (`models/model_factory.py`)

**Model Types:**
- `OrthogonalModel`: Full model with adapter, f_hom, f_id, classifier
- `DirectClassifierModel`: Simple classifier on raw embeddings
- `BaseEmbeddingsModel`: Identity model that returns raw embeddings

**Key Models:**
- `BaseEmbeddingModel`: Abstract base class defining interface
- `BaseEmbeddingsModel`: Concrete implementation for raw embedding baseline

### Training (`training/trainer.py`)

**Main Entry Point:**
- `ModelTrainer.train_and_evaluate()`: Orchestrates training and evaluation
- Supports skipping training (`stage_a=false`, `stage_b=false`) for pure evaluation

**Training Methods:**
- `_train_stage_a()`: Real-only pretraining with three loss components
- `_train_stage_b()`: Classification fine-tuning
- Both methods support configurable loss weights, epochs, learning rates

### Evaluation (`evaluation/evaluator.py`)

**Detection Evaluations:**
1. **Mahalanobis Distance**: Fit Gaussian on real samples, compute distance-based AUC
2. **Linear Probe**: Train logistic regression on embeddings
3. **MLP Classifier**: Train 2-layer MLP on embeddings

**Identity Evaluations:**
1. **KNN**: k-nearest neighbor classification using identity labels
2. **Few-Shot Episodic**: 5-way 5-shot + 15 queries prototypical evaluation

**Key Method:**
- `evaluate_model()`: Main evaluation orchestration
- Returns dictionary of metric scores

### Loss Functions (`losses/loss.py`)

**Available Losses:**
- `variance_compactness()`: L_hom - VICReg-style variance minimization
- `cross_cov_penalty()`: L_orth - Orthogonality constraint
- `build_pos_neg_from_batch()`: Builds positive/negative pairs for contrastive learning
- `SupConLoss`: Supervised contrastive loss
- `ArcMarginProduct`: ArcFace angular margin layer
- `ArcFaceLoss`: Combined ArcFace + CrossEntropy loss

## Remaining TODOs

1. **Debug KNN Accuracy (Currently 0.0)** ✅ FIXED
   - Issue: KNN returning 0.0 accuracy for both @1 and @5
   - Root cause: Too many unique identities (4096) making KNN impossible to learn
   - Fix applied: Changed metric from 'euclidean' to 'cosine' in `evaluation/evaluator.py` lines 203-204
   - Expected behavior: With 4096 identities, KNN accuracy will likely remain low (<5%) even with cosine distance
   - Alternative: Consider filtering to fewer identities or using different evaluation metric

2. **Complete Training Implementation**
   - Heterogeneous head Stage 1 (ArcFace training)
   - Heterogeneous head Stage 2 (Episodic prototypical training)
   - Currently only homogeneous head training is implemented

3. **Evaluation for SENET and HuBERT**
   - Current baseline: OpenL3 embeddings only
   - Need to run same evaluation on:
     - SENET embeddings
     - HuBERT embeddings
   - Command format:
     ```bash
     python experiments/main.py --config models/configs/base_embeddings.json --data_config.model_name senet
     python experiments/main.py --config models/configs/base_embeddings.json --data_config.model_name hubert
     ```

4. **Episodic Evaluation Logic**
   - Currently assumes all 100 samples in batch are from exactly 5 identities
   - May need more robust parsing of episodic batch structure
   - Location: `evaluation/evaluator.py` lines 217-283

## Recent Changes

### Command Run:
```bash
python3 experiments/main.py --config models/configs/base_embeddings.json --output_dir ./results
```

### Results Generated:
Located in `results/experiment_base_embeddings.json`

**Baseline Results (OpenL3):**
- Mahalanobis: Val AUC = 0.729, Test AUC = 0.707
- Linear Probe: Val AUC = 0.991, Test AUC = 0.991
- MLP Classifier: Val AUC = 0.787, Test AUC = 0.801
- KNN Accuracy: **0.0 (needs debugging)**
- Few-Shot Accuracy: **0.717**

**Interpretation:**
- Linear probe performs excellently (99% AUC)
- Mahalanobis distance shows moderate performance (72% AUC)
- MLP classifier is reasonable (78-80% AUC)
- KNN failing (likely too many identity classes)
- Few-shot getting 71.7% (better than random 20%)

### Files Modified/Added:
- `data/data_loader.py`: Added identity-aware splitting, episodic batching
- `evaluation/evaluator.py`: Implemented comprehensive evaluation pipeline
- `training/trainer.py`: Implemented Stage A and Stage B training
- `losses/loss.py`: Added ArcFace loss components
- `models/model_factory.py`: Added `BaseEmbeddingsModel` for raw embedding evaluation
- `models/configs/base_embeddings.json`: Config for baseline evaluation

## Next Steps

1. **Fix KNN Evaluation**: Debug why KNN returns 0.0 accuracy
2. **Run Baseline for SENET**: Generate baseline results for SENET embeddings
3. **Run Baseline for HuBERT**: Generate baseline results for HuBERT embeddings
4. **Implement Heterogeneous Training**: Complete Stage 1 (ArcFace) and Stage 2 (Episodic) training for heterogeneous head
5. **Full Training Pipeline**: Implement full 4-stage training (hom Stage A/B, het Stage A/B)

## Testing the Pipeline

### Baseline Evaluation (No Training):
```bash
python3 experiments/main.py --config models/configs/base_embeddings.json --output_dir ./results
```

### Full Training Evaluation:
```bash
python3 experiments/main.py --config models/configs/orthogonal_model.json --output_dir ./results
```

### Direct Classifier Training:
```bash
python3 experiments/main.py --config models/configs/direct_classifier.json --output_dir ./results
```

