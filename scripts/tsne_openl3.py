"""
t-SNE visualization of OpenL3 segment embeddings.

Loads per-segment OpenL3 embeddings from the HDF5 file (avdeepfake1m only),
labels each segment as real (< 0.5) or fake (>= 0.5), then plots a t-SNE
with green = real, purple = fake.

Usage:
    python -m scripts.tsne_openl3
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from datetime import datetime

HDF5_PATH  = "exports/deepfake_embeddings.h5"
RESULTS_DIR = os.path.join("results", "tsne")
MAX_SEGS   = 20000   # cap to keep t-SNE tractable
SEED       = 42

# ── Load segments ─────────────────────────────────────────────────────────
print("Loading OpenL3 segment embeddings (avdeepfake1m)...")

real_embeddings = []
fake_embeddings = []

with h5py.File(HDF5_PATH, "r") as f:
    for vk in f["videos"].keys():
        vid = f["videos"][vk]
        ds = vid.attrs.get("dataset", b"")
        if isinstance(ds, bytes):
            ds = ds.decode()
        if ds != "avdeepfake1m":
            continue

        aug_types = [
            t.decode() if isinstance(t, bytes) else t
            for t in vid["augmentation_info"]["types"][:]
        ]

        for aug_idx, aug_type in enumerate(aug_types):
            emb          = vid["embeddings"]["openl3"][aug_idx]   # (T, 512)
            audio_labels = vid["labels"]["audio"][aug_idx]        # (T,)

            if aug_type in ("source", "real"):
                for seg_emb in emb:
                    real_embeddings.append(seg_emb)
            elif aug_type == "fake":
                for seg_emb in emb:
                    fake_embeddings.append(seg_emb)

real_embeddings = np.array(real_embeddings, dtype=np.float32)
fake_embeddings = np.array(fake_embeddings, dtype=np.float32)

# Assign labels: 0.0 = real, 1.0 = fake
embeddings = np.concatenate([real_embeddings, fake_embeddings], axis=0)
labels     = np.concatenate([
    np.zeros(len(real_embeddings), dtype=np.float32),
    np.ones(len(fake_embeddings),  dtype=np.float32),
])

embeddings = np.array(embeddings, dtype=np.float32)
labels     = np.array(labels,     dtype=np.float32)

print(f"Total segments: {len(embeddings):,}  "
      f"(real: {(labels < 0.5).sum():,}, fake: {(labels >= 0.5).sum():,})")

# ── Subsample if needed ───────────────────────────────────────────────────
if len(embeddings) > MAX_SEGS:
    rng = np.random.default_rng(SEED)
    # Balanced subsample
    real_idx = np.where(labels < 0.5)[0]
    fake_idx = np.where(labels >= 0.5)[0]
    n_each   = min(MAX_SEGS // 2, len(real_idx), len(fake_idx))
    sel = np.concatenate([
        rng.choice(real_idx, n_each, replace=False),
        rng.choice(fake_idx, n_each, replace=False),
    ])
    embeddings = embeddings[sel]
    labels     = labels[sel]
    print(f"Subsampled to {len(embeddings):,} segments ({n_each:,} real + {n_each:,} fake)")

# ── t-SNE ─────────────────────────────────────────────────────────────────
print("Running t-SNE (this may take a few minutes)...")
tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=SEED, verbose=1)
coords = tsne.fit_transform(embeddings)

# ── Plot ──────────────────────────────────────────────────────────────────
colors = np.where(labels < 0.5, "green", "purple")

fig, ax = plt.subplots(figsize=(10, 8))
for label_val, color, marker_label in [(0, "green", "Real"), (1, "purple", "Fake")]:
    mask = (labels < 0.5) if label_val == 0 else (labels >= 0.5)
    ax.scatter(
        coords[mask, 0], coords[mask, 1],
        c=color, s=12, alpha=0.5, linewidths=0, label=f"{marker_label} ({mask.sum():,})",
    )

ax.set_title("t-SNE of OpenL3 Segment Embeddings (avdeepfake1m)", fontsize=14, fontweight="bold")
ax.set_xlabel("t-SNE dim 1")
ax.set_ylabel("t-SNE dim 2")
ax.legend(markerscale=4, fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

os.makedirs(RESULTS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(RESULTS_DIR, f"tsne_openl3_{timestamp}.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved to: {save_path}")