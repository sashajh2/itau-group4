# Loss Balancing Alternatives

## The Problem with Additive Combination

The current approach:
```
L_total = L_proto + λ_var * L_var + λ_orth * L_orth
```

**Issues:**
1. Fixed weights (λ) are arbitrary and hard to tune
2. Loss magnitudes differ, so one loss dominates
3. No guarantee that all losses contribute meaningfully

## Alternative Approaches

### 1. **Gradient Balancing (GradNorm)** ✅ Implemented
- **How it works**: Automatically adjusts weights to balance gradient magnitudes
- **Pros**: Principled, automatic balancing
- **Cons**: More complex, requires gradient computation during forward pass
- **Best for**: When you want automatic balancing without manual tuning

### 2. **Alternating Optimization** (Simpler Alternative)
- **How it works**: Optimize one loss at a time, cycling through them
- **Pros**: Simple, no weights needed, each loss gets full attention
- **Cons**: Slower convergence, might oscillate
- **Best for**: When losses are truly independent

### 3. **Equal-Weight Normalization** (Simplest)
- **How it works**: Normalize each loss by its initial value, then use equal weights
- **Pros**: Very simple, no hyperparameters
- **Cons**: Assumes initial losses are representative
- **Best for**: Quick experiments, baseline

### 4. **Uncertainty Weighting** (Kendall et al. 2018)
- **How it works**: Learn task uncertainty as part of the model
- **Pros**: Very principled, learns optimal weights
- **Cons**: Adds parameters, more complex
- **Best for**: When you have compute and want best results

## Recommendation

For your use case, I recommend **Gradient Balancing** (already implemented) because:
1. It automatically balances contributions
2. No manual weight tuning needed
3. Prevents any loss from collapsing
4. More principled than fixed weights

But if you want something simpler, **Alternating Optimization** is also good - it's what you suggested: "minimize the 2 projections separately, and then also try to minimize the loss of the orthogonality constraint"

