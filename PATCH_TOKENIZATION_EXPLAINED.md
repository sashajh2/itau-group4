# Patch Tokenization: Deep Dive

## 1. Is Patch Tokenization Ideal for Deepfake Detection?

### **Pros:**
✅ **Efficiency**: Reduces sequence length (T segments → N patches, where N < T)
   - Example: 100 segments → ~25 patches (with patch_size=8, stride=4)
   - Attention complexity: O(N²) instead of O(T²) - much faster!

✅ **Local Pattern Capture**: Groups consecutive segments that likely share similar characteristics
   - Deepfake artifacts often appear in short bursts (e.g., lip-sync errors, audio glitches)
   - Patches naturally capture these local anomalies

✅ **Temporal Context**: Overlapping patches ensure continuity
   - Segment 5 appears in both Patch 0 and Patch 1
   - Prevents boundary artifacts

✅ **Proven Architecture**: PatchTST (Time Series Transformer) shows strong performance on temporal tasks

### **Cons:**
❌ **Information Loss**: Mean pooling within patches loses fine-grained segment-level details
   - If only 1 segment in a patch is fake, averaging might dilute the signal

❌ **Fixed Patch Size**: May not match natural temporal boundaries
   - Deepfake transitions might not align with patch boundaries

### **Verdict:**
**Yes, it's a good choice**, but with caveats:
- Works well for capturing **temporal patterns** and **long-range dependencies**
- May lose some **fine-grained segment-level** information
- Consider experimenting with different `patch_size` values (4, 8, 16)

---

## 2. Does It Model Both Local and Global Context?

### **Yes! Here's how:**

### **Local Context (Intra-Patch):**
```
Patch 0: [s0, s1, s2, s3, s4, s5, s6, s7]
          ↓ [Mean Pooling]
     Single vector representing local pattern
```

- **Before attention**: Segments within a patch are aggregated (mean pooled)
- This captures **local temporal patterns** (e.g., 8 consecutive segments)
- The aggregation happens **before** the Transformer, so local info is preserved in the patch representation

### **Global Context (Inter-Patch):**
```
Patch 0 ──┐
Patch 1 ──┤
Patch 2 ──┼──→ [Self-Attention] ──→ Each patch attends to ALL other patches
Patch 3 ──┤
Patch 4 ──┘
```

- **During attention**: Each patch can attend to all other patches
- This captures **long-range dependencies** (e.g., early fake segment → later real segment)
- The Transformer encoder (4 layers) allows patches to exchange information globally

### **Example:**
```
Video with fake segment at position 10:
  Patch 0: [s0-s7]   (real)
  Patch 1: [s4-s11]   (contains fake at s10) ← Local: mean pooling captures anomaly
  Patch 2: [s8-s15]   (real)
  
After attention:
  - Patch 1 knows it's different (local context)
  - Patch 0 and Patch 2 can "see" Patch 1's anomaly (global context)
  - All patches get contextualized based on full sequence
```

**So yes, you get both:**
- **Local**: Mean pooling within patches captures short-term patterns
- **Global**: Self-attention between patches captures long-range dependencies

---

## 3. How Do We Attend Between Patches?

### **Patches are Mean-Pooled Tensors**

Looking at the code:

```python
# In AVTemporalModel.__init__:
use_mean_pooling = config.patch_size > 1  # True when patch_size > 1

# In PatchTokenizer.forward():
if self.use_mean_pooling:
    # Mean pool over patch dimension: (B, N, P, d) -> (B, N, d)
    patches = patches.mean(dim=2)
```

**What this means:**
- Input segments: `(B, T=20, d=512)`
- After patching: `(B, N=5, P=8, d=512)` - 5 patches, each with 8 segments
- After mean pooling: `(B, N=5, d=512)` - Each patch is now a **single vector**

**Attention happens on these patch vectors:**

```python
# In ModalityEncoder.forward():
patches: (B, N=5, d=512)  # Each patch is a single vector
  ↓ [Patch Projection]
projected: (B, N=5, 256)  # Project to model dimension
  ↓ [Positional Encoding]
positioned: (B, N=5, 256)  # Add position info
  ↓ [Transformer Encoder]
encoded: (B, N=5, 256)  # Self-attention between N=5 patch vectors
```

**Attention Mechanism:**
- Each of the 5 patch vectors attends to all 5 patch vectors (including itself)
- This is standard **self-attention** in Transformers
- Query, Key, Value are all computed from the patch vectors

**Visual:**
```
Patch 0 vector ──┐
Patch 1 vector ──┤
Patch 2 vector ──┼──→ [Self-Attention: Q, K, V from patch vectors]
Patch 3 vector ──┤
Patch 4 vector ──┘
```

**Key Point:**
- A patch is a **single tensor/vector** (after mean pooling)
- Attention is between these patch vectors, not between individual segments
- This is why it's efficient: 5 patches vs 20 segments

---

## 4. How Does Upsampling Work?

### **The Problem:**
- Model outputs: `patch_logits (B, N=5)` - one logit per patch
- We need: `segment_logits (B, T=20)` - one logit per segment

### **The Solution:**
Upsampling maps patch logits back to segments using the `segment_to_patches` mapping.

### **Step-by-Step:**

#### **Step 1: Build Mapping During Tokenization**
```python
# In PatchTokenizer.forward():
segment_to_patches = []
for seg_idx in range(T):  # For each segment (0 to 19)
    covering_patches = []
    for patch_idx in range(N):  # For each patch (0 to 4)
        patch_start = patch_idx * patch_stride
        patch_end = patch_start + patch_size
        if patch_start <= seg_idx < patch_end:
            covering_patches.append(patch_idx)
    segment_to_patches.append(covering_patches)
```

**Example with patch_size=8, stride=4:**
```
Segment 0: covered by Patch 0 only → [0]
Segment 1: covered by Patch 0 only → [0]
Segment 2: covered by Patch 0 only → [0]
...
Segment 4: covered by Patch 0 AND Patch 1 → [0, 1]  ← Overlap!
Segment 5: covered by Patch 0 AND Patch 1 → [0, 1]
...
Segment 7: covered by Patch 0 AND Patch 1 → [0, 1]
Segment 8: covered by Patch 1 AND Patch 2 → [1, 2]
...
```

#### **Step 2: Upsample Using Mapping**
```python
# In upsample_patch_logits():
segment_logits = torch.zeros(B, T)  # Initialize

for seg_idx in range(T):
    covering_patches = segment_to_patches[seg_idx]  # Which patches cover this segment?
    patch_logits_for_seg = patch_logits[:, covering_patches]  # Get logits from those patches
    
    if method == "average":
        segment_logits[:, seg_idx] = patch_logits_for_seg.mean(dim=1)  # Average them
    elif method == "max":
        segment_logits[:, seg_idx] = patch_logits_for_seg.max(dim=1)[0]  # Or take max
```

### **Concrete Example:**

**Input:**
- 20 segments, patch_size=8, stride=4
- Creates 5 patches: [0-7], [4-11], [8-15], [12-19], [16-19]
- Patch logits: `[0.8, 0.3, 0.9, 0.7, 0.6]` (one per patch)

**Upsampling:**
```
Segment 0: covered by Patch 0 → logit = 0.8
Segment 1: covered by Patch 0 → logit = 0.8
...
Segment 4: covered by Patch 0 AND Patch 1 → logit = (0.8 + 0.3) / 2 = 0.55
Segment 5: covered by Patch 0 AND Patch 1 → logit = (0.8 + 0.3) / 2 = 0.55
...
Segment 8: covered by Patch 1 AND Patch 2 → logit = (0.3 + 0.9) / 2 = 0.6
...
```

**Result:**
- Each segment gets a logit
- Overlapping segments average logits from multiple patches
- This smooths predictions and handles boundary cases

### **Why This Works:**
1. **Overlap Handling**: Segments covered by multiple patches get averaged logits
2. **Smoothness**: Averaging creates smoother predictions across boundaries
3. **Information Preservation**: All patch-level information is distributed to segments

### **Alternative: Max Pooling**
If you use `method="max"`:
- Segment 4: `max(0.8, 0.3) = 0.8` (takes highest confidence)
- More aggressive, preserves strong signals

---

## Summary

1. **Is patch tokenization ideal?** 
   - ✅ Yes, for efficiency and pattern capture
   - ⚠️ But may lose fine-grained details

2. **Local + Global context?**
   - ✅ Yes: Mean pooling captures local, attention captures global

3. **How do we attend?**
   - Patches are mean-pooled vectors `(B, N, d)`
   - Self-attention operates between these patch vectors

4. **Upsampling?**
   - Maps patch logits → segment logits using `segment_to_patches` mapping
   - Overlapping segments average logits from multiple covering patches

