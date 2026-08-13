---
marp: true
title: Disentangled Representation Losses
paginate: true
---

# Disentangled Representation Learning

**Three losses, one goal: separate _content_ from _authenticity_**

Each audio sample is encoded into two independent embedding spaces:

- **`z_id`** — *identity / content* (what is being said)
- **`z_auth`** — *authenticity* (real vs. spoofed/fake)

The losses keep these two axes clean and independent.

---

# The Big Picture

| Loss | Operates on | Enforces |
|---|---|---|
| `L_proto` | `z_id` | content forms meaningful clusters |
| `L_var` | `z_auth` (real only) | "real" is one tight cluster → anomaly boundary |
| `L_orth` | `z_id` vs `z_auth` | the two spaces stay independent |

$$
L_{total} = L_{proto} + \lambda_{var} \cdot L_{var} + \lambda_{orth} \cdot L_{orth}
$$

---

# `L_proto` — Prototypical Contrastive Loss

**Job: organize the identity space `z_id` by content.**

- Compute a **prototype** per content group = mean embedding of the group
- Pull each sample toward its own prototype, push it from all others
- Softmax over negative Euclidean distances, scaled by temperature $\tau$

$$
L_{proto} = -\frac{1}{N} \sum_i \log \frac{\exp(-d(z_i, c_{correct})/\tau)}{\sum_k \exp(-d(z_i, c_k)/\tau)}
$$

**Effect:** same-content samples cluster tightly; different content separates.

---

# `L_var` — Variance Minimization Loss

**Job: collapse all *real* samples into one tight region of `z_auth`.**

- Take only real samples, compute their centroid $\mu_{real}$
- Minimize squared distance of each real sample to that centroid

$$
L_{var} = \frac{1}{N_{real}} \sum_i \| z_i^{auth} - \mu_{real} \|^2
$$

**Intuition:** *there are many ways to be fake, but "real" is one coherent thing.*
Real samples form a compact cluster → fakes fall outside → clean anomaly boundary.

> A regularization term prevents trivial collapse (everything → one point).
> ⚠️ Currently disabled by default: `lambda_var = 0`.

---

# `L_orth` — Orthogonality Constraint

**Job: keep `z_id` and `z_auth` independent.**

- Without it, authenticity info could leak into the content axis
- Penalize correlation by driving cosine similarity toward zero

$$
L_{orth} = \frac{1}{N^2} \sum_{i,j} \left| \cos\text{-}sim(z_i^{id}, z_j^{auth}) \right|
$$

**Low `L_orth` ⇒ near-orthogonal axes ⇒ truly disentangled.**
Changing *what's said* shouldn't move the *real/fake* decision, and vice versa.

---

# Why Balancing Matters

The three losses:

- pull the model in **different directions**
- have **different magnitudes**
- decrease at **different rates**

Without balancing, one loss dominates and the others never contribute.

→ This motivates the **multi-objective balancing** strategies
(fixed weights, EMA normalization, equal-weight, GradNorm).

---

# Summary

- **`L_proto`** structures the *content* space `z_id`
- **`L_var`** makes *real* a tight cluster in `z_auth` for anomaly detection
- **`L_orth`** keeps the two spaces independent (disentangled)

Together they let the model **judge authenticity independently of content.**
