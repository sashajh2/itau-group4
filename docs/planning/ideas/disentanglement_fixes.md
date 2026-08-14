# Disentanglement Training: Failure Analysis & Fixes

## Observed failures (from checkpoint runs)

| Run | Epochs | var (final) | orth (final) | sil_gt (final) | sil_clusters |
|---|---|---|---|---|---|
| `disentangled` (baseline) | 50 | **6e-8 (collapsed)** | 0.0003 | -0.019 | 0.074 |
| `disentangled_equal_weights` | 50 | 0.051 | 0.002 | 0.057 | **0.697** |
| `disentangled_improved_v1` | 6 | 0.052 | 0.001 | 0.082 | 0.669 |
| `disentangled_improved_v2` | 50 | 0.052 | 0.0004 | 0.090 (peak 0.127 @ ep7) | 0.684 |

**Key observations:**
- z_id learns identity structure well in all non-baseline runs (sil_clusters 0.67–0.70)
- z_auth separation of real/fake is weak at best (sil_gt 0.06–0.09) and negative in the baseline
- `improved_v2` peaks at sil_gt=0.127 at epoch 7 then **degrades** — orth loss has decayed by then (0.028 → 0.0004), letting z_id bleed back into z_auth
- The baseline variance collapse happens in ~7 epochs: var 0.0023 → 6e-8

---

## Root cause: no gradient signal for fakes in z_auth

The `variance_loss` is:

```
L_var = (1/N_real) * Σ ||z_i^auth - μ_real||²  +  regularization
```

**Fakes have zero gradient from this loss.** Their position in `z_auth` is determined only by:
1. `orthogonality_loss` — wants `z_auth ⊥ z_id`, says nothing about real vs fake separation
2. Initialization + whatever proto bleeds through the orth constraint

Result: fakes drift close to the real centroid in `z_auth` space, not because the model is confused, but because nothing is pushing them away.

---

## Fix 1: Add fake repulsion to `variance_loss`

### The problem in detail

`mu_real` is the mean of real embeddings in `z_auth` space. The attraction term pulls all reals toward this centroid. Without a repulsion term, the optimal position for a fake is **anywhere** — and the shortest path gradient from `orthogonality_loss` or proto bleed tends to place them near the reals.

### Variant 1a — Hard margin repulsion (recommended starting point)

Add a hinge loss that pushes each fake at least `margin` distance from the real centroid:

```python
def variance_loss(z_auth, is_real, min_variance=0.01, regularization_weight=0.1,
                  repulsion_margin=0.5, lambda_repel=1.0):

    z_auth_real = z_auth[is_real]
    z_auth_fake = z_auth[~is_real]

    if z_auth_real.shape[0] < 2:
        return torch.tensor(0.0, device=z_auth.device, requires_grad=True)

    # Attraction: pull reals toward centroid
    mu_real = z_auth_real.mean(dim=0, keepdim=True)  # [1, emb_dim]
    variance = ((z_auth_real - mu_real) ** 2).sum(dim=1).mean()

    # Anti-collapse regularization (unchanged)
    reg_quad = torch.clamp(min_variance - variance, min=0.0) ** 2
    reg_lin  = torch.clamp(min_variance - variance, min=0.0)
    regularization = reg_quad + 5.0 * reg_lin

    loss = variance + regularization_weight * regularization

    # Repulsion: push fakes away from real centroid
    if z_auth_fake.shape[0] >= 1:
        mu_real_detached = mu_real.detach()  # CRITICAL: don't pull centroid toward fakes
        dist_fake = ((z_auth_fake - mu_real_detached) ** 2).sum(dim=1).sqrt()  # L2 distance
        repulsion = F.relu(repulsion_margin - dist_fake).mean()  # hinge: 0 if far enough
        loss = loss + lambda_repel * repulsion

    return loss
```

**Why `mu_real.detach()` is critical:** without it, the repulsion gradient backpropagates through `mu_real`, which is a function of the real embeddings. This would pull the real centroid *toward* the fakes to reduce the distance — the opposite of what we want.

**Hyperparameter guidance:**
- `repulsion_margin`: embeddings are L2-normalized (unit sphere), so distances range 0–2. A margin of **0.5–1.0** is reasonable. Start at 0.5.
- `lambda_repel`: start at **1.0** (same scale as attraction). If fakes still cluster near reals after 10 epochs, increase to 2.0.

**Expected behavior:** variance and repulsion act as a push-pull on the real/fake geometry in `z_auth`. Reals cluster tightly; fakes are pushed to the periphery.

---

### Variant 1b — Unconstrained distance maximization (no hyperparameter, but unbounded)

```python
if z_auth_fake.shape[0] >= 1:
    mu_real_detached = mu_real.detach()
    dist_fake = ((z_auth_fake - mu_real_detached) ** 2).sum(dim=1).sqrt()
    repulsion = -dist_fake.mean()  # maximize distance, no margin
    loss = loss + lambda_repel * repulsion
```

**Pro:** no margin hyperparameter, always has a gradient.  
**Con:** unbounded — fakes can be pushed arbitrarily far, wasting representational capacity. Proto loss may fight back. Use only if 1a shows no movement.

---

### Variant 1c — Two-prototype auth loss (most principled, requires stable fake batches)

Replace the scalar centroid with a full binary prototypical classification in `z_auth` space:

```python
def auth_prototypical_loss(z_auth, is_real, temperature=0.5):
    """Binary prototype loss: classify each segment as real or fake using z_auth."""
    if is_real.sum() < 2 or (~is_real).sum() < 2:
        return torch.tensor(0.0, device=z_auth.device, requires_grad=True)

    proto_real = z_auth[is_real].mean(dim=0)          # [emb_dim]
    proto_fake = z_auth[~is_real].mean(dim=0)          # [emb_dim]
    prototypes = torch.stack([proto_real, proto_fake])  # [2, emb_dim]

    # Negative L2 distance to each prototype
    distances = -torch.cdist(z_auth, prototypes, p=2)  # [batch, 2]
    logits = distances / temperature
    targets = is_real.long()  # 1=real→class0, 0=fake→class1; flip: targets = (~is_real).long()
    # real=class 0, fake=class 1
    targets = (~is_real).long()

    return F.cross_entropy(logits, targets)
```

Then in `compute_total_loss`, replace `L_var` with `auth_prototypical_loss` (or run both).

**Pro:** both real and fake samples get explicit gradient signal; uses all label information; directly optimizes for separability.  
**Con:** requires at least 2 fakes per batch reliably. At batch_size=128 with a ~50/50 split this is fine, but can break with small batches or heavy class imbalance. The fake prototype is noisy with few samples — consider using a momentum-updated prototype (EMA across batches) for stability.

---

### Variant 1d — EMA prototype for stability across small batches

If fake batches are small or class imbalance is high, single-batch prototypes are noisy. Use an exponential moving average:

```python
class EMAPrototype:
    def __init__(self, dim=128, momentum=0.99):
        self.proto_real = None
        self.proto_fake = None
        self.momentum = momentum

    def update(self, z_auth, is_real):
        with torch.no_grad():
            if is_real.sum() > 0:
                batch_real = z_auth[is_real].mean(dim=0)
                self.proto_real = (batch_real if self.proto_real is None
                                   else self.momentum * self.proto_real + (1 - self.momentum) * batch_real)
            if (~is_real).sum() > 0:
                batch_fake = z_auth[~is_real].mean(dim=0)
                self.proto_fake = (batch_fake if self.proto_fake is None
                                   else self.momentum * self.proto_fake + (1 - self.momentum) * batch_fake)

    def repulsion_loss(self, z_auth, is_real, margin=0.5):
        if self.proto_real is None:
            return torch.tensor(0.0, device=z_auth.device, requires_grad=True)
        z_auth_fake = z_auth[~is_real]
        if z_auth_fake.shape[0] == 0:
            return torch.tensor(0.0, device=z_auth.device, requires_grad=True)
        dist = ((z_auth_fake - self.proto_real) ** 2).sum(dim=1).sqrt()
        return F.relu(margin - dist).mean()
```

Call `ema.update(z_auth.detach(), is_real)` at the end of each batch, then use `ema.repulsion_loss(z_auth, is_real)` in the loss.

---

## Fix 2: Replace orthogonality loss with cross-covariance regularizer

Current `orthogonality_loss` computes per-sample pairwise cosine similarities and clamps to a minimum. It decays to near-zero by epoch 7–10 as the model learns to be orthogonal — then there's no ongoing enforcement.

**Replace with a cross-covariance penalty (VICReg-style):**

```python
def cross_covariance_loss(z_id: torch.Tensor, z_auth: torch.Tensor) -> torch.Tensor:
    """
    Penalize cross-covariance between z_id and z_auth across the batch.
    Unlike pairwise cosine sim, this stays non-zero even after per-sample orthogonality
    is achieved because it measures distributional correlation.
    """
    N, D = z_id.shape

    # Center each representation
    z_id_c   = z_id   - z_id.mean(dim=0)
    z_auth_c = z_auth - z_auth.mean(dim=0)

    # Cross-covariance matrix: [D, D]
    cross_cov = (z_id_c.T @ z_auth_c) / (N - 1)

    # Penalize all entries (both diagonal and off-diagonal)
    loss = (cross_cov ** 2).sum() / D

    return loss
```

**Why this is better:** The existing orth loss computes `(1/N²) Σ |cos(z_id_i, z_auth_j)|`. Once individual vectors are orthogonal, this sum is ~0. The cross-covariance matrix measures whether the *dimensions* of z_id correlate with *dimensions* of z_auth across the whole dataset — it stays non-zero even when individual samples are orthogonal.

**Also: apply stop-gradient on z_id in orth computation:**

```python
# In compute_total_loss:
L_orth = cross_covariance_loss(z_id.detach(), z_auth)
```

This means only `f_auth` gets gradients from the orth term. `f_id` is not penalized — it can learn identity freely while `f_auth` is pushed to be decorrelated from it. This removes the gradient conflict where orth was simultaneously pulling both heads in opposite directions.

---

## Fix 3: VICReg variance term to prevent dimensional collapse

The current regularization prevents *sample* collapse (all samples to one point) but not *dimensional* collapse (many dimensions go to zero, reducing effective capacity).

```python
def vicreg_variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    Prevent dimensional collapse: penalize dimensions with std < gamma.
    Apply to both z_auth and z_id.
    """
    std = torch.sqrt(z.var(dim=0) + eps)             # [emb_dim]
    loss = F.relu(gamma - std).mean()                 # penalize dims with low variance
    return loss
```

Apply to z_auth in `compute_total_loss`:
```python
L_dim_collapse = vicreg_variance_loss(z_auth)
total_loss = ... + lambda_dim * L_dim_collapse
```

Start with `lambda_dim=0.5` and `gamma=1.0` (for unit-normalized embeddings, std≈1 is healthy).

---

## Recommended experiment order

1. **Start with Fix 1a + Fix 2 (stop-gradient only)** — minimum code change, addresses both root causes.
   - Add `mu_real.detach()` and the repulsion hinge to `variance_loss`
   - Change `L_orth = orthogonality_loss(z_id.detach(), z_auth)` in `compute_total_loss`
   - Expected: sil_gt should exceed 0.15 and not degrade after epoch 7

2. **If 1a doesn't move fakes (check sil_gt after 5 epochs):** switch to Variant 1c (two-prototype auth loss) — it gives fakes a direct gradient from day 1.

3. **If runs still degrade long-term:** add Fix 2 (cross-covariance loss) to replace the decaying orth loss.

4. **Add Fix 3 (VICReg variance)** as a safety net in any run — low cost, prevents silent dimensional collapse.

---

## What's already working (don't break it)

- `prototypical_contrastive_loss` is doing its job well (sil_clusters 0.67–0.70 in all non-baseline runs)
- `EqualWeightNormalizer` successfully prevents the var collapse seen in the baseline
- The `improved_v2` run shows the model *can* reach sil_gt≈0.13 early — the architecture has capacity, the training signal just degrades

The goal is to keep z_id healthy (sil_clusters staying above 0.65) while pushing z_auth's sil_gt above 0.15 sustainably across 50 epochs.
