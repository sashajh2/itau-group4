# Fix-1a (Hard-Margin Fake Repulsion) — Results & Explanation

**Experiment:** `experiments/fix1a_repulsion_experiment.py`
**Data:** AVDeepFake1M, 20% stratified subset (by video), HuBERT embeddings
**Config:** 20 epochs · `batch_size=128` · `lr=1e-4` · `repulsion_margin=0.5` · `lambda_repel=1.0` · `seed=42`
**Segment counts:** 59,792 real / 1,389 fake (≈ 98.9% / 1.1%)

---

## 1. What Fix-1a does

Fix-1a is the **baseline disentangler plus one extra loss term** — a *hard-margin repulsion* that
pushes fake embeddings away from the real cluster's centroid. Nothing else changes.

### Baseline (inherited) setup

The model is `DisentangledProjector`, which splits each input into two latent vectors:

- **`z_id`** — content/identity code
- **`z_auth`** — authenticity code (the real-vs-fake space)

The baseline trains with three losses, balanced by an `EqualWeightNormalizer`:

1. **proto** — prototypical contrastive loss on `z_id` (groups same-content samples)
2. **var** (attraction) — pulls *real* samples' `z_auth` toward the real centroid `μ_real`
3. **orth** — keeps `z_id` and `z_auth` orthogonal (disentangled)

Nothing in the baseline ever tells fake samples where to go — so it collapses: real and fake
embeddings pile onto a single point (all pairwise cosine sims ≈ 0.9997).

### The one term Fix-1a adds

From `experiments/fix1a_repulsion_experiment.py:227-233`:

```python
mu_real_sg = mu_real.detach()                          # real centroid, stop-gradient
dist_fake  = ((z_fake - mu_real_sg) ** 2).sum(1).sqrt()  # ‖z_fake − μ_real‖₂
repulsion_loss = F.relu(repulsion_margin - dist_fake).mean()   # margin = 0.5
```

- Compute each **fake** sample's Euclidean distance to the **real** centroid.
- `relu(0.5 − dist)` penalizes any fake *closer* than the margin. A fake already ≥ 0.5 away
  contributes zero; a fake sitting on top of the real cluster contributes the full penalty.
- Minimizing this **pushes fakes outward** until they clear the margin.

Two deliberate choices:

- **`mu_real.detach()` (stop-gradient):** repulsion moves only the *fake* embeddings, not the
  real centroid — it can't cheat by dragging the real cluster around instead of separating fakes.
- **Self-normalized:** `L_repel` is divided by its own initial value and weighted by
  `λ_repel = 1.0`, so it starts at scale ~1.0 alongside the three normalized losses.

**In one sentence:** Fix-1a = baseline three-loss disentangler + a hinge penalty
`relu(0.5 − ‖z_fake − μ_real.detach()‖₂).mean()` that shoves fake authenticity-embeddings at
least a margin of 0.5 away from the (fixed) real cluster.

---

## 2. How we tackled class imbalance

AVDeepFake1M is heavily skewed at the segment level: **≈ 98.9% real / 1.1% fake** (the fakes are
partial audio fakes inside otherwise-real videos). Three mechanisms address this:

1. **`WeightedRandomSampler` (batch balancing)** — `experiments/fix1a_repulsion_experiment.py:543-548`.
   Each class is weighted by the inverse of its frequency
   (`class_weights = 1 / count`), so real and fake segments are drawn into batches at roughly
   equal rates despite the 98.9/1.1 split. Sampling is with replacement, so the rare fakes are
   oversampled rather than the common reals being discarded.

   ```python
   class_counts   = [n_real, n_fake]
   class_weights  = [1.0 / max(c, 1) for c in class_counts]
   sample_weights = [class_weights[l] for l in train_labels]
   sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
   ```

2. **Video-level stratified split** — train/val are split by *video* (`VAL_FRAC=0.20`), not by
   segment, so segments from one video never leak across the split and the natural real/fake
   proportion is preserved in each partition.

3. **Repulsion is robust to sparse fakes** — even with the sampler, a batch of 128 still contains
   only a handful of fakes. The repulsion term uses `.mean()` over whatever fakes are present and
   simply returns 0 when a batch has none (`:228-229`), so the signal accumulates over steps
   rather than requiring dense fake coverage per batch.

> Note: the class balancing shapes the *batches*; the natural imbalance is still what the raw
> metrics reflect. This is why the repulsion effect is real but modest.

---

## 3. Results — Fix-1a vs. Baseline

Final-epoch (epoch 20) values. "Raw" = untrained encoder embeddings, for reference.

| Metric | Raw | Baseline | Fix-1a | Δ (1a − base) |
|---|---:|---:|---:|---:|
| **separation_gap** | −0.0023 | 0.0000 | **0.0086** | +0.0086 |
| cos sim real→real | 0.7739 | 0.9997 | 0.9057 | −0.0940 |
| cos sim fake→real | 0.7762 | 0.9997 | 0.8971 | −0.1026 |
| cos sim fake→fake | 0.7866 | 0.9997 | 0.9005 | −0.0992 |
| KL divergence | 2.5475 | 1.7916 | 3.3334 | +1.5418 |
| JS distance | 0.3553 | 0.3245 | 0.3329 | +0.0084 |
| Wasserstein | 0.2285 | 0.0017 | 0.0222 | +0.0205 |
| variability_ratio | 0.9683 | 0.9373 | 0.8723 | −0.0650 |

### Final training losses

| Loss term | Baseline | Fix-1a |
|---|---:|---:|
| total | 0.2455 | 1.3968 |
| proto | 0.0073 | 0.0066 |
| var   | 0.0052 | 0.2068 |
| orth  | 0.0129 | 0.0386 |
| repel | 0.0000 | 0.0012 |

---

## 4. How `separation_gap` is calculated

From `training/disentangled/metrics.py:234`:

```
separation_gap = mean_cos_sim(real → real_centroid) − mean_cos_sim(fake → real_centroid)
```

1. **Split** embeddings into real (label 0) and fake (label 1).
2. **Real centroid** = mean of all real embeddings, then L2-normalized.
3. **L2-normalize** every individual embedding.
4. **real→real** = mean cosine similarity of each real sample to the real centroid.
5. **fake→real** = mean cosine similarity of each fake sample to the *real* centroid.
6. **Gap** = (real→real) − (fake→real).

**Intuition:** how much more tightly real samples cluster around the real centroid than fakes do.
A **positive** gap means fakes sit farther from the real manifold (good separation); **zero** means
reals and fakes are indistinguishable from the real centroid's perspective. It is a one-sided,
centroid-based measure (only the real centroid is the reference — it does not check whether fakes
cluster among *themselves*), so it is directional rather than a full two-cluster separation score.

---

## 5. Readout

The baseline **collapses** the representation — every pairwise cosine similarity sits at ~0.9997
and `separation_gap` is exactly 0, so real and fake are identical. Fix-1a's repulsion term breaks
that collapse: cosine sims drop to ~0.90 with a small but **positive** real/fake separation
(+0.0086). The cost is a larger `var` loss. Fix-1a fixes the *collapse symptom*, but with only
~1.1% fakes the separation it buys is real yet modest.
