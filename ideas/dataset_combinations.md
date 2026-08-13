# Dataset Combination Strategies

## What we have

| Dataset       | Real videos | Fake videos | Notes |
|---------------|-------------|-------------|-------|
| AVDeepFake1M  | 241         | 121         | Audio manipulation (speech synthesis/conversion) |
| ShareVeo3     | 0           | 1460        | Different manipulation type (likely face/lip-sync) |
| Sora2         | ?           | ?           | AI-generated video (generative model artifacts) |

Key structural facts:
- All real samples come from AVDeepFake1M (the only source of real videos)
- ShareVeo3 fakes dominate the fake pool (1460 vs 121)
- Duration and num_segments are stored per video in the HDF5 attrs

---

## Combination 1 — AVDeepFake1M only (short vs long video split)

Split AVDeepFake1M videos by duration rather than by dataset:
  - Short: videos with num_segments ≤ median → train set
  - Long:  videos with num_segments > median → test set

**Why:** Tests whether the model generalizes across video lengths — a common
failure mode for sequence models that overfit to sequence length distributions.
Longer videos have more segments, more context, and may expose different artifacts.

**What to expect:** If the model relies on seeing a full fake audio segment,
short clips may be harder to classify. If artifacts are local, length shouldn't matter.

**How to run:** Filter the HDF5 scan by `vid.attrs["num_segments"]` to build
separate short/long pools. Use the standard sampling function on each pool.

---

## Combination 2 — Train on ShareVeo3 fakes, test on AVDeepFake1M fakes (cross-dataset zero-shot)

Train: 300 real (AVDeepFake1M) + 300 fake (ShareVeo3 only)
Test:  100 real (AVDeepFake1M) + 100 fake (AVDeepFake1M only)

**Why:** This is the hardest and most honest test of generalization. If the model
learns something general about "what a deepfake sounds/looks like" rather than
"what a ShareVeo3 fake sounds like", it will transfer. If it doesn't transfer,
the features are dataset-specific.

**What to expect:** Poor performance — this is expected and informative. The FNR
on AVDeepFake1M fakes will reveal how much of our current "success" was ShareVeo3
pattern memorization.

**Interpretation guide:**
  FNR ≈ 0.5 → model is guessing randomly on AVDeepFake1M fakes (no transfer)
  FNR < 0.3  → meaningful transfer exists despite the domain gap
  FNR > 0.7  → model actively inverts on new domain (ShareVeo3 artifacts anti-correlate)

---

## Combination 3 — Train on AVDeepFake1M fakes, test on ShareVeo3 fakes (reverse cross-dataset)

Train: 300 real (AVDeepFake1M) + ~100 fake (AVDeepFake1M, all we have for train)
Test:  100 real (AVDeepFake1M) + 100 fake (ShareVeo3 only)

**Why:** Tests the reverse direction. AVDeepFake1M has only 121 fake videos —
far fewer than ShareVeo3. This reflects a realistic scenario where you have a
small labeled set of one fake type and must detect a different type in deployment.

**Limitation:** Only ~91 fake training videos available after the 75/25 split.
This will require careful sampling to avoid over-representation.

---

## Combination 4 — Stratified by manipulation type (recommended next step)

Enforce equal representation across fake types in both train and test:
  - Within AVDeepFake1M: audio-only manipulation vs video+audio manipulation
    (check augmentation_info for sub-type labels if available)
  - Across datasets: 50% AVDeepFake1M + 50% ShareVeo3 (already done)

Extension: if Sora2 data is available, add a three-way stratification:
  100 fake from AVDeepFake1M + 100 fake from ShareVeo3 + 100 fake from Sora2

**Why:** Generative model artifacts (Sora2) are qualitatively different from
face-swap or voice-conversion artifacts. A robust detector should handle all three.

---

## Combination 5 — Leave-one-dataset-out cross-validation

Three folds:
  Fold 1: train on {AVDeepFake1M, Sora2}, test on ShareVeo3
  Fold 2: train on {ShareVeo3, Sora2}, test on AVDeepFake1M
  Fold 3: train on {AVDeepFake1M, ShareVeo3}, test on Sora2

**Why:** The most rigorous evaluation of cross-dataset generalization. Each
dataset gets to be the held-out test set exactly once. The variance in
performance across folds reveals which source is hardest to generalize to/from.

**What this answers:** "Which dataset is most different from the others?" and
"Is there a training combination that produces a universally robust detector?"

---

## Combination 6 — Balanced real source + multi-fake (production-realistic)

Train: AVDeepFake1M real + [equal parts AVDeepFake1M fake, ShareVeo3 fake]
Test:  Held-out AVDeepFake1M real + held-out fakes from ALL sources

This is closest to a real-world detector that must handle fakes from unknown
sources. The test set deliberately includes all fake types so no single type dominates.

**Why:** Currently our test sets are also imbalanced (only two datasets, one of
which dominates). A truly representative test set should mirror deployment conditions.

---

## Recommended priority order

| Priority | Combination | Answers |
|----------|-------------|---------|
| 1 | Combination 2 (ShareVeo3→AVDeepFake1M zero-shot) | How much of our results are dataset-specific? |
| 2 | Combination 4 (three-way stratified with Sora2)   | Can one model handle all fake types? |
| 3 | Combination 5 (leave-one-out CV)                  | Which fake type is hardest to detect? |
| 4 | Combination 1 (short vs long within AVDeepFake1M) | Does video length affect detection? |
| 5 | Combination 3 (AVDeepFake1M→ShareVeo3 zero-shot)  | Does audio-trained model detect visual fakes? |
| 6 | Combination 6 (production-realistic)              | End-to-end deployment readiness |

---

## Note on real video scarcity

All real videos come from AVDeepFake1M (241 total). This is a hard constraint —
if we want to test cross-dataset generalization on the real side, we are limited.
One option is to use augmentation (pitch shift, time stretch, noise injection) on
real audio to artificially expand the real pool, though this introduces its own biases.
