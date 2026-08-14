# Embedding Combination Strategies

We have three pre-computed embeddings per video segment:
  - HuBERT  (768-dim)  — audio speech representation
  - OpenL3  (512-dim)  — audio/music representation (environment-aware)
  - SENet   (2048-dim) — video visual representation

The single-embedding stratified results:
  HuBERT:  acc=0.635, FNR=0.200
  OpenL3:  acc=0.755, FNR=0.460
  SENet:   acc=0.665, FNR=0.340

None dominates on both metrics. Combination is the natural next step.

---

## Strategy 1 — Early fusion (concatenation)

Concatenate all three embeddings at the segment level before the model:
  [hubert | openl3 | senet] → 768+512+2048 = 3328-dim per segment

Feed the 3328-dim sequence directly into the transformer or MLP.

**Pros:** Simplest to implement, already done in earlier experiments (train.py).
**Cons:** The model must learn to disentangle audio vs visual signal from scratch.
  High-dimensional input may require more data.

**Variants to try:**
- hubert + openl3 only (3328 → 1280, audio-only multimodal)
- hubert + senet only (3328 → 2816, audio + video)
- all three (3328-dim)

**Hypothesis:** hubert + senet should outperform any single embedding since
AVDeepFake1M has audio manipulation (→ hubert) and ShareVeo3 likely has
visual artifacts (→ senet).

---

## Strategy 2 — Late fusion (ensemble predictions)

Train three independent models (one per embedding), then combine their output
probabilities at inference time:

  p_final = w_h · p_hubert + w_o · p_openl3 + w_s · p_senet

Weights can be:
  - Equal (1/3 each) — simple average
  - Learned on a validation set — logistic regression on [p_h, p_o, p_s]
  - Performance-weighted: weight by (1 - FNR) on a held-out set

**Pros:** Each model trains on its own embedding independently, no architecture changes.
  Best single-modality knowledge is preserved.
**Cons:** Three separate training runs, no cross-modal interaction.

**Why this is worth trying:** The three embeddings make different error patterns.
HuBERT has low FNR (misses few fakes) but lower accuracy. OpenL3 has high precision
but high FNR. A weighted ensemble could combine HuBERT's recall with OpenL3's
precision.

---

## Strategy 3 — Intermediate fusion (two-stream with cross-attention)

Encode audio and video separately to a shared dimension, then fuse:

  audio_seq  = Encoder_A([hubert | openl3])   → (T, d_model)
  visual_seq = Encoder_V(senet)                → (T, d_model)

  fused = CrossAttention(query=audio_seq, key=visual_seq, value=visual_seq)
  z     = Pool(fused) → classifier

**Pros:** Allows the model to learn which visual frames correspond to suspicious
audio, enabling A/V consistency checking (a key deepfake signal).
**Cons:** More complex, requires more data to train the cross-attention weights.

**Why this matters:** A/V sync inconsistency is one of the most reliable deepfake
signals — the lip movement doesn't match the audio. Cross-attention can model this
explicitly if the segments are temporally aligned.

---

## Strategy 4 — Modality dropout (robustness regularization)

During training, randomly zero out one entire modality per sample (e.g. 30% of
the time, set all senet embeddings to zero; 30% of the time, zero out audio).

The model learns to detect fakes from any subset of available modalities.

**Pros:** Forces the model to not rely on any single embedding's shortcut.
  Produces a model that is robust to missing modalities at test time.
**Cons:** Requires careful tuning of dropout rates.

**Why this matters:** In production, video or audio tracks may be corrupted or
unavailable. Modality dropout trains robustness to this explicitly.

---

## Strategy 5 — Learned modality weighting (attention over embeddings)

Instead of concatenating, learn a soft attention weight over the three embeddings
at each time step:

  w_t = softmax(MLP([h_t, o_t, s_t]))   # 3 weights per segment
  e_t = w_t[0]·h_t + w_t[1]·o_t + w_t[2]·s_t

Then feed the weighted combination e_t to the classifier.

**Pros:** The model learns which modality is most informative per segment.
  Interpretable — you can visualize which modality the model trusts most.
**Cons:** Adds parameters; may overfit on small datasets.

---

## Recommended experimental order

| Priority | Combination        | Expected effort | Expected gain |
|----------|--------------------|-----------------|---------------|
| 1        | Early fusion (hubert+senet) | Low      | High — covers both manipulation types |
| 2        | Late fusion ensemble       | Low       | Medium — free improvement via diversity |
| 3        | All three concatenated     | Low       | Medium — already partially tested |
| 4        | Modality dropout           | Medium    | Medium — regularization benefit |
| 5        | Cross-attention fusion     | High      | High if A/V sync is the key signal |
| 6        | Learned modality weighting | Medium    | Low-medium — likely overfits |
