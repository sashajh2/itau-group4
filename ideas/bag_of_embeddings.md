# Bag of Embeddings Approaches

The transformer and LSTM experiments assume temporal order matters — but our results
suggest the models struggle to exploit it (or it may not matter much for this task).
A bag-of-embeddings approach treats each video as an unordered set of segment
embeddings, which is both simpler and more regularized.

---

## Why this might work better

- Deepfake artifacts (codec artifacts, voice inconsistency, spectral anomalies) may
  appear uniformly across segments rather than in a specific temporal pattern
- With only 300 training videos, sequence models have limited data to learn meaningful
  positional dependencies
- Bag approaches tend to generalize better across video lengths

---

## Approach 1 — Aggregate statistics + MLP (simplest baseline)

Compute a fixed-length descriptor per video by pooling across all T segments:
- Concatenate [mean, std, max, min] across the time axis → 4×D vector
- Feed into a small MLP (2–3 layers) for binary classification

**Why:** Extremely low parameter count, no risk of sequence overfitting.
Good sanity check — if this beats the transformer, temporal modeling is not helping.

Implementation: ~30 lines of code, no DataLoader needed, runs in seconds.

---

## Approach 2 — Attention-weighted pooling (no positional encoding)

Instead of mean pooling, learn a scalar attention weight per segment:
  a_t = softmax(W · h_t)   →   z = Σ a_t · h_t

The attention is content-based (what does the segment look like?) rather than
position-based. A single linear layer produces the weight logits.

**Why:** Gives the model flexibility to focus on the most anomalous segments
without committing to a temporal order hypothesis.

Variant: use two-level attention (segment-level then video-level) for interpretability
— you can inspect which segments drove the decision.

---

## Approach 3 — DeepSets

A theoretically grounded permutation-invariant architecture:
  z = ρ(Σ φ(h_t))

where φ is a per-element MLP and ρ is an aggregation MLP.
Sum pooling after φ is the key — it is provably universal over set functions.

**Why:** More expressive than mean pooling while still being order-agnostic.
Efficient and easy to implement (two small MLPs).

---

## Approach 4 — Multiple Instance Learning (MIL)

Treat each video as a "bag" and each segment as an "instance."
The bag label is positive if any instance is positive (weak supervision).

Standard approach: ABMIL (Attention-Based MIL)
  - Each segment gets an attention score via a gated mechanism
  - The bag representation is the attention-weighted sum of instances
  - A linear head on the bag representation produces the final prediction

**Why:** MIL is the natural framework when segment-level labels are unreliable
or when you only care about video-level detection. It also produces segment-level
anomaly scores as a free byproduct — useful for interpretability.

Paper reference: Ilse et al., "Attention-Based Deep Multiple Instance Learning" (ICML 2018)

---

## Approach 5 — NetVLAD / Fisher Vector encoding

Encode the distribution of segment embeddings rather than their order:
- Fit K Gaussian clusters (e.g. K=16 or 64) on the training embeddings
- For each video, compute a residual encoding: how much does each segment
  deviate from each cluster center?
- Concatenate the residuals → fixed K×D vector, regardless of video length

**Why:** Captures the "vocabulary" of embedding patterns rather than their sequence.
NetVLAD was originally designed for place recognition but works well for any
variable-length set of feature vectors. Can be trained end-to-end or used with a
fixed k-means initialization.

---

## Approach 6 — Random forest / gradient boosting on segment statistics

Non-neural baseline using handcrafted features per video:
- Per-dimension: mean, std, skewness, kurtosis, min, max, 25th/75th percentile
- Cross-dimension: top principal components of the segment covariance matrix
- Feed to XGBoost or LightGBM

**Why:** Highly interpretable, immune to gradient instability, and often competitive
with neural approaches on small datasets (<1000 training samples).
If this performs comparably to the transformer, the problem is data-limited, not
architecture-limited.

---

## Recommended order to try

1. Aggregate stats + MLP (establish a strong baseline quickly)
2. Attention-weighted pooling (most likely to win — simple but expressive)
3. ABMIL (if segment-level interpretability is valuable)
4. DeepSets (if ABMIL doesn't help)
5. NetVLAD (if distribution-level encoding is worth exploring)
6. Gradient boosting (as a sanity check on the neural approaches)
