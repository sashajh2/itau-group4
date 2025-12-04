# Embedding Collapse Analysis & Recommendations

## Problem Diagnosis

From your metrics history and evaluation results:

### **Symptoms:**
1. **Variance Loss Value**: Stays constant (~0.05) but this is misleading
2. **Actual Embedding Variance**: Collapses dramatically
   - Wasserstein distance: 0.533 → 0.002 (-99.7%)
   - Mean distances: 5.7 → 0.23 (-96%)
   - Intra-group variance: 13.78 → 0.02 (-99.8%)

### **Root Causes:**

1. **Weak Regularization**: `variance_reg_weight=0.1` is too weak
   - Regularization term: `0.1 * regularization` is dominated by main variance term
   - When variance → 0, regularization gradient is too small to prevent collapse

2. **Low Minimum Variance Threshold**: `min_variance=0.01` is too low
   - Allows embeddings to collapse to very small region
   - Should be higher to maintain meaningful separation

3. **Batch-Level Variance**: Variance loss computed per-batch
   - Global variance can collapse even if batch variance stays above threshold
   - No global constraint on embedding spread

4. **L2 Normalization**: Forces all embeddings onto unit sphere
   - Reduces effective embedding space
   - Can contribute to collapse when combined with variance minimization

5. **No Explicit Separation**: Only minimizes real variance, doesn't push fakes away
   - Real samples cluster tightly (good)
   - But fake samples also collapse to same region (bad)
   - Need explicit separation loss

---

## Recommended Fixes

### **Fix 1: Strengthen Variance Regularization** ⭐ (High Priority)

**Current:**
```python
min_variance = 0.01
variance_reg_weight = 0.1
```

**Recommended:**
```python
min_variance = 0.1  # 10x increase
variance_reg_weight = 1.0  # 10x increase
```

**Why:**
- Higher threshold forces embeddings to maintain spread
- Stronger regularization weight ensures gradient signal when variance is low
- Prevents collapse while still allowing variance minimization

### **Fix 2: Add Explicit Separation Loss** ⭐⭐⭐ (Critical)

**New Loss Component:**
```python
def separation_loss(z_auth: torch.Tensor, is_real: torch.Tensor, 
                   margin: float = 0.5) -> torch.Tensor:
    """
    Push fake samples away from real centroid.
    
    Args:
        z_auth: Authenticity embeddings
        is_real: Boolean mask
        margin: Minimum distance between real and fake centroids
    
    Returns:
        scalar loss
    """
    z_auth_real = z_auth[is_real]
    z_auth_fake = z_auth[~is_real]
    
    if z_auth_real.shape[0] < 2 or z_auth_fake.shape[0] < 2:
        return torch.tensor(0.0, device=z_auth.device, requires_grad=True)
    
    # Compute centroids
    mu_real = z_auth_real.mean(dim=0)
    mu_fake = z_auth_fake.mean(dim=0)
    
    # Distance between centroids
    centroid_distance = torch.norm(mu_real - mu_fake)
    
    # Penalize if distance is less than margin
    separation_penalty = torch.clamp(margin - centroid_distance, min=0.0) ** 2
    
    return separation_penalty
```

**Add to total loss:**
```python
L_sep = separation_loss(z_auth, is_real, margin=0.5)
total_loss = L_proto + lambda_var * L_var + lambda_orth * L_orth + lambda_sep * L_sep
```

**Why:**
- Explicitly enforces separation between real and fake
- Prevents both classes from collapsing to same region
- Margin parameter controls minimum separation distance

### **Fix 3: Make L2 Normalization Optional** ⭐⭐ (Medium Priority)

**Current:** Always L2-normalizes outputs

**Recommended:** Add flag to disable normalization
```python
class DisentangledProjector(nn.Module):
    def __init__(self, ..., normalize_outputs: bool = False):
        self.normalize_outputs = normalize_outputs
    
    def forward(self, z):
        z_auth = self.f_auth(z)
        z_id = self.f_id(z)
        
        if self.normalize_outputs:
            z_auth = F.normalize(z_auth, dim=-1)
            z_id = F.normalize(z_id, dim=-1)
        
        return z_auth, z_id
```

**Why:**
- L2 normalization constrains embeddings to unit sphere
- Removing it allows embeddings to use full space
- Can help prevent collapse

### **Fix 4: Add Global Variance Monitoring** ⭐ (Low Priority)

**Add to training loop:**
```python
# Compute global variance (not just batch)
with torch.no_grad():
    all_z_auth_real = []
    for batch in dataloader:
        z_auth, _ = model(batch['embeddings'])
        all_z_auth_real.append(z_auth[batch['is_real']])
    
    all_z_auth_real = torch.cat(all_z_auth_real)
    global_variance = ((all_z_auth_real - all_z_auth_real.mean(dim=0)) ** 2).sum(dim=1).mean()
    print(f"Global variance: {global_variance.item():.6f}")
```

**Why:**
- Batch variance can be misleading
- Global variance shows true collapse
- Helps diagnose issues early

### **Fix 5: Increase Gradient Clipping** (Low Priority)

**Current:** `gradient_clip = 1.0`

**Recommended:** `gradient_clip = 5.0` or `10.0`

**Why:**
- Stronger regularization might need larger gradients
- Prevents over-clipping of important signals

---

## Implementation Priority

1. **Immediate (Do First):**
   - Fix 1: Increase `min_variance` and `variance_reg_weight`
   - Fix 2: Add separation loss

2. **Next:**
   - Fix 3: Make L2 normalization optional
   - Fix 4: Add global variance monitoring

3. **Optional:**
   - Fix 5: Adjust gradient clipping

---

## Expected Results

After fixes:
- **Wasserstein distance**: Should stay > 0.1 (not collapse to 0.002)
- **Mean distances**: Should stay > 1.0 (not collapse to 0.23)
- **Separation gap**: Should be positive and meaningful (> 0.1)
- **Variance**: Should stay above `min_variance` threshold

---

## Hyperparameter Ranges to Try

```python
# Conservative (start here)
min_variance = 0.1
variance_reg_weight = 1.0
separation_margin = 0.5
lambda_sep = 0.5

# Moderate
min_variance = 0.2
variance_reg_weight = 2.0
separation_margin = 1.0
lambda_sep = 1.0

# Aggressive (if still collapsing)
min_variance = 0.5
variance_reg_weight = 5.0
separation_margin = 2.0
lambda_sep = 2.0
```

