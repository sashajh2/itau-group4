# Experiment Summary

## Phase 0 — Embedding Analysis (pre-existing)

**Cosine similarity & linear separability per embedding (AVDeepFake1M, ~4.6K real / ~6K fake segments)**

| Embedding | Cosine distance (real vs fake) | Linear separability |
|-----------|-------------------------------|---------------------|
| HuBERT    | 0.279 (high)                  | 82.2%               |
| OpenL3    | 0.005 (near zero)             | 82.1%               |
| SENet     | 0.028 (low)                   | 67.5%               |

Key tension: OpenL3 has almost no cosine separation between real and fake, yet its
linear probe matches HuBERT. This suggests the real/fake signal in OpenL3 is there
but encoded differently — not in the direction of the mean.

**OpenL3 full evaluation** (Mahalanobis, linear probe, MLP, few-shot):
- Linear probe AUC: **0.991** — exceptionally high
- MLP AUC: 0.840
- Mahalanobis AUC: 0.707
- Few-shot (5-way 5-shot): 0.708

---

## Phase 1 — Sequence Model Baselines (pre-existing)

**LSTM — HuBERT only** (no dataset filter, ~400 train / 200 test, 75 epochs with early stopping)
- Performed well — no collapse

**LSTM — Concatenated (HuBERT + OpenL3 + SENet, 3328-dim)** (same setup)
- Results in `results/lstm/`

**Transformer — 5-fold and 3-fold k-fold CV** (HuBERT, multiple runs, checkpoints saved)
- Extensive cross-validation with saved `.pt` checkpoints

**Transformer — Cross-dataset** (train on AVDeepFake1M, test on ShareVeo3, HuBERT)
- Results in `results/transformer/`

---

## Phase 2 — Single Embedding Ablation

**Experiment A — AVDeepFake1M only, no stratification** (`train_single_embedding.py`)
- 800 samples (600 train / 200 test), Transformer, all three embeddings
- **Result: collapse** — HuBERT and OpenL3 predicted all-fake; SENet barely learned
- **Finding:** AVDeepFake1M has only 121 fake videos → 30 in test pool → 3.3x repetition
  → model finds the "predict all fake" shortcut (F1=0.667 beats any noisy partial solution)

**Experiment B — All datasets, no stratification** (`train_all_datasets.py`)
- 800 samples (600 train / 200 test), Transformer, HuBERT / SENet / OpenL3
- **Result: ~95% accuracy, FNR ~0.07–0.09 across all embeddings**
- **Finding: Illusory performance** — dataset breakdown revealed 278/300 fake train
  samples came from ShareVeo3, so the model learned ShareVeo3 signatures, not
  deepfakes in general

**Experiment C — Stratified 50/50 fake split** (`train_stratified.py`)
- 800 samples (600 train / 200 test), Transformer, HuBERT / OpenL3 / SENet
- Forced 150 fakes per dataset (AVDeepFake1M + ShareVeo3)
- **Results (test):**

| Embedding | Accuracy | FNR   | F1    |
|-----------|----------|-------|-------|
| HuBERT    | 0.635    | 0.200 | 0.687 |
| OpenL3    | 0.755    | 0.460 | 0.688 |
| SENet     | 0.665    | 0.340 | 0.663 |

- **Finding:** True performance after removing dataset shortcut. HuBERT has the
  lowest FNR. All models overfit to train (99%+ train accuracy) while test plateaus
  at 63–75%. The gap between Experiment B and C quantifies exactly how much of the
  earlier "success" was dataset artifact.

---

## Phase 3 — Positional Encoding Ablation

**Experiment D** (`compare_pos_encoding.py`)
- 600 samples (400 train / 200 test), AVDeepFake1M only
- Vanilla Transformer (with pos enc) vs Transformer (no pos enc)
- HuBERT and OpenL3
- Metrics: accuracy, FNR, ROC-AUC
- Results pending

---

## Structural observations for a paper

Four natural angles depending on what the paper argues:

1. **"Dataset bias is the dominant confound"** — Experiments B vs C tell a clean story
   about how uncontrolled fake pool composition inflates reported performance. This is
   a methodological contribution.

2. **"HuBERT is the most reliable single embedding for audio deepfake detection"** —
   Phase 0 (cosine similarity), Phase 2 (lowest FNR in stratified setting), and prior
   LSTM results all consistently point the same way.

3. **"Positional encoding vs permutation invariance"** — Experiment D directly tests
   whether temporal order carries useful information. If the no-pos-enc model matches
   or beats the standard transformer, it motivates the bag-of-embeddings direction in
   `bag_of_embeddings.md`.

4. **"The case for cross-dataset evaluation as a standard"** — Experiments A through C
   together argue that within-dataset and unbalanced-pool evaluation are systematically
   misleading, and propose stratified cross-dataset evaluation as the correct protocol.
