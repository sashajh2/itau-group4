# Transformer Model Walkthrough: Understanding `time_series_model.py`

## Overview

The `AVTemporalModel` is a **PatchTST-style temporal Transformer** that processes **multimodal (audio + video) embeddings** across time to detect deepfakes. It's currently designed as an **AV fusion model** (requires both modalities), but can be adapted for single-modality use.

---

## Architecture Flow: Step-by-Step

### **Input Stage: Raw Embeddings**
```
Input: 
  - audio_seq: (B, T, d_a) where d_a = 512 (openl3) or 768 (hubert)
  - video_seq: (B, T, d_v) where d_v = 2048 (senet)
  
B = batch size
T = number of temporal segments (varies per video)
d_a = audio embedding dimension
d_v = video embedding dimension
```

### **Step 1: Patch Tokenization** (`PatchTokenizer`)
Converts temporal sequences into overlapping patches (like Vision Transformers do with image patches).

```python
# Example with patch_size=8, patch_stride=4
# Input: (B, T=20, d=512) audio sequence
# Output: (B, N=5, d=512) patches (with mean pooling)
#         OR (B, N=5, 8*512=4096) flattened patches

N = number of patches = ceil((T - patch_size) / patch_stride) + 1
```

**What it does:**
- Groups consecutive segments into patches (e.g., segments 0-7 → patch 0, segments 4-11 → patch 1, etc.)
- Creates overlapping patches for better temporal coverage
- With `use_mean_pooling=True`, averages segments within each patch → `(B, N, d)`
- Without pooling, flattens patches → `(B, N, patch_size*d)`

**Why patches?**
- Reduces sequence length (T segments → N patches, where N < T)
- Captures local temporal patterns
- Makes attention computation more efficient

### **Step 2: Modality Encoding** (`ModalityEncoder`)
Each modality (audio/video) gets its own Transformer encoder.

```python
# For each modality:
patches: (B, N, patch_dim) 
  ↓ [Patch Projection]
projected: (B, N, model_dim=256)
  ↓ [+ Positional Encoding]
positioned: (B, N, 256)
  ↓ [Transformer Encoder: 4 layers, 8 heads]
encoded_tokens: (B, N, 256)
```

**Components:**
1. **Patch Projection**: Linear layer `(patch_dim → model_dim)` to project patches into common dimension
2. **Positional Encoding**: Learnable embeddings added to each patch position
3. **Transformer Encoder**: 
   - 4 layers of self-attention + feedforward
   - 8 attention heads
   - Model dimension = 256
   - Processes patches independently per modality

**Output:** Contextualized tokens `(B, N, 256)` for each modality

### **Step 3: Fusion** (`FusionHead`)
Combines audio and video tokens into a single representation.

```python
audio_tokens: (B, N, 256)
video_tokens: (B, N, 256)
  ↓ [Concatenate]
fused: (B, N, 512)  # 256 + 256
  ↓ [Fusion MLP: 512 → 512 → 256 → 1]
patch_logits: (B, N)  # Per-patch real/fake logits
```

**What it does:**
- Concatenates audio + video tokens along feature dimension
- Passes through MLP to produce per-patch logits
- Each patch gets a single logit (real/fake score)

### **Step 4: Upsampling** (`upsample_patch_logits`)
Converts patch-level predictions back to segment-level.

```python
patch_logits: (B, N)  # e.g., N=5 patches
  ↓ [Map patches → segments using segment_to_patches]
segment_logits: (B, T)  # e.g., T=20 segments
```

**How it works:**
- Each segment may be covered by multiple overlapping patches
- For each segment, averages (or maxes) logits from all covering patches
- Result: One logit per temporal segment

### **Final Output**
```python
segment_logits: (B, T)  # Per-segment real/fake logits
patch_info: Dict  # Metadata about patches
```

---

## Current Architecture: AV Fusion Only

**Yes, this is strictly an AV fusion model** in its current form. It requires:
- Both `audio_seq` and `video_seq` inputs
- Both `audio_encoder` and `video_encoder`
- `FusionHead` that concatenates both modalities

**Can you train on just audio?** 
- **Not directly** - the model expects both inputs
- **But you can modify it** - see adaptation strategies below

---

## Adapting for Disentangled Embeddings

### **Your Disentangled Model Outputs:**
From `DisentangledProjector`:
- **z_auth**: (B, 128) - Authenticity embeddings (L2-normalized)
- **z_id**: (B, 128) - Identity embeddings (L2-normalized)

**Current status:** You've trained this for **audio (Hubert)** embeddings only.

### **Option 1: Audio-Only Transformer (Recommended First Step)**

Create a single-modality version that uses only `z_auth` embeddings:

```python
class AudioTemporalModel(nn.Module):
    """Single-modality temporal transformer for audio authenticity embeddings."""
    
    def __init__(self, config):
        # Only audio components
        self.audio_tokenizer = PatchTokenizer(...)
        self.audio_encoder = ModalityEncoder(...)
        # No fusion - direct classification head
        self.classifier = nn.Linear(model_dim, 1)
    
    def forward(self, audio_seq):
        # audio_seq: (B, T, 128) - z_auth embeddings over time
        patches, patch_info = self.audio_tokenizer(audio_seq)
        tokens = self.audio_encoder(patches)  # (B, N, 256)
        patch_logits = self.classifier(tokens).squeeze(-1)  # (B, N)
        segment_logits = upsample_patch_logits(patch_logits, patch_info)
        return segment_logits
```

**Advantages:**
- Simpler architecture
- Can test if audio-only temporal modeling works
- No need for video disentanglement yet

### **Option 2: AV Fusion with Disentangled Embeddings**

If you want to keep AV fusion, you'll need:

1. **Audio disentangled embeddings** ✅ (you have this)
2. **Video disentangled embeddings** ❌ (you need to train this)

**Steps:**
1. Train `DisentangledProjector` on video (SENET) embeddings
2. Extract `z_auth` for both audio and video
3. Modify `AVTemporalModel` to accept `z_auth` embeddings instead of raw embeddings

**Modified dataset:**
```python
class DisentangledAVH5Dataset(Dataset):
    def __getitem__(self, idx):
        # Load raw embeddings
        audio_raw = vid_grp["embeddings/hubert"][aug_idx]
        video_raw = vid_grp["embeddings/senet"][aug_idx]
        
        # Project through disentangled models
        audio_z_auth = audio_disentangled_model(audio_raw)  # (T, 128)
        video_z_auth = video_disentangled_model(video_raw)  # (T, 128)
        
        return {
            "audio_seq": audio_z_auth,  # (T, 128) instead of (T, 768)
            "video_seq": video_z_auth,  # (T, 128) instead of (T, 2048)
            "label_seq": labels,
            ...
        }
```

---

## Key Design Decisions

### **Why Patches?**
- **Efficiency**: Reduces sequence length from T segments to N patches
- **Local patterns**: Captures short-term temporal dependencies
- **Overlap**: Ensures continuity between patches

### **Why Separate Encoders?**
- **Modality-specific processing**: Audio and video have different temporal characteristics
- **Flexibility**: Can use different architectures per modality (though currently same)
- **Interpretability**: Can analyze each modality separately

### **Why Fusion?**
- **Complementary information**: Audio and video provide different signals
- **Robustness**: Can handle cases where one modality is unreliable
- **Performance**: Multimodal fusion typically outperforms single-modality

---

## Input/Output Summary

### **Inputs:**
```python
audio_seq: torch.Tensor  # (B, T, d_a)
  - Raw audio embeddings (openl3: 512, hubert: 768)
  - OR disentangled z_auth: (B, T, 128)
  
video_seq: torch.Tensor  # (B, T, d_v)
  - Raw video embeddings (senet: 2048)
  - OR disentangled z_auth: (B, T, 128)
  
valid_length: Optional[torch.Tensor]  # (B,)
  - Actual sequence lengths (for padding handling)
```

### **Outputs:**
```python
segment_logits: torch.Tensor  # (B, T)
  - Per-segment real/fake logits
  - Higher = more likely real
  - Apply sigmoid for probabilities
  
patch_info: Dict
  - Metadata about patch structure
  - Used for upsampling
```

---

## Recommendations

### **For Your Use Case:**

1. **Start with Audio-Only**:
   - Modify `AVTemporalModel` to work with just audio `z_auth` embeddings
   - Test if temporal modeling improves over segment-level classification
   - Simpler to debug and faster to train

2. **If Audio-Only Works Well:**
   - Consider training video disentanglement
   - Then implement AV fusion with disentangled embeddings
   - Compare: audio-only vs. AV fusion performance

3. **Dataset Modification:**
   - Create a new dataset class that:
     - Loads raw embeddings from HDF5
     - Runs them through your trained `DisentangledProjector` (audio)
     - Returns `z_auth` sequences instead of raw embeddings
   - Update `ModelConfig` to use `audio_emb_dim=128` (instead of 512/768)

---

## Code Changes Needed

### **Minimal Changes for Audio-Only:**

1. **New Model Class:**
   ```python
   class AudioTemporalModel(nn.Module):
       # Remove video components
       # Keep only audio_tokenizer, audio_encoder
       # Replace FusionHead with simple Linear classifier
   ```

2. **New Dataset:**
   ```python
   class DisentangledAudioDataset(Dataset):
       def __init__(self, hdf5_path, disentangled_model_path, ...):
           # Load disentangled model
           self.disentangled_model = load_disentangled_model(...)
           
       def __getitem__(self, idx):
           # Load raw embeddings
           raw_emb = vid_grp["embeddings/hubert"][aug_idx]
           # Project to z_auth
           z_auth = self.disentangled_model(torch.tensor(raw_emb))
           return {"audio_seq": z_auth, ...}
   ```

3. **Update Config:**
   ```python
   config = ModelConfig(
       audio_emb_dim=128,  # z_auth dimension
       # Remove video_emb_dim
   )
   ```

---

## Summary

- **Current model**: AV fusion (requires both modalities)
- **Your disentangled model**: Audio-only (z_auth from Hubert)
- **Can train audio-only transformer**: Yes, with modifications
- **Need video disentanglement**: Only if you want AV fusion
- **Recommended path**: Start with audio-only, then add video if needed

The transformer architecture is modular enough that you can easily adapt it for single-modality use or disentangled embeddings!

