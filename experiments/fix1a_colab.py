"""
Fix 1a: Hard-margin fake repulsion — self-contained Google Colab script.
No repo dependencies. Loads pre-exported .npz file.

─── Setup ────────────────────────────────────────────────────────────────────
Step 1 (local):  python scripts/export_embeddings_npz.py
                 → exports/avdeepfake_20pct_embeddings.npz  (~70 MB)

Step 2 (Colab):  Upload the .npz, then run this script.
                 !pip install -q scikit-learn scipy tqdm matplotlib
                 (torch is pre-installed on Colab)

Step 3:          Set NPZ_PATH below and click Runtime → Run all
─────────────────────────────────────────────────────────────────────────────
"""

# ── 0. Colab setup (uncomment as needed) ─────────────────────────────────────
# from google.colab import files
# uploaded = files.upload()          # click "Choose Files" and select the .npz
# NPZ_PATH = list(uploaded.keys())[0]

# — OR mount Google Drive —
# from google.colab import drive
# drive.mount('/content/drive')
# NPZ_PATH = '/content/drive/MyDrive/avdeepfake_20pct_embeddings.npz'

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

# ── 2. Config ─────────────────────────────────────────────────────────────────
NPZ_PATH         = "avdeepfake_20pct_embeddings.npz"   # ← set this path
REPULSION_MARGIN = 0.50
VAL_FRAC         = 0.20
BATCH_SIZE       = 128
NUM_EPOCHS       = 20
LR               = 1e-4
WEIGHT_DECAY     = 1e-5
WARMUP_STEPS     = 500
TEMPERATURE      = 0.1
MIN_VARIANCE     = 0.01
VAR_REG_WEIGHT   = 0.1
MIN_ORTH         = 0.001
LAMBDA_REPEL     = 1.0
SEED             = 42
SAVE_DIR         = "results_fix1a"
MAX_EVAL_SAMPLES = 5000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 3. Data ───────────────────────────────────────────────────────────────────

def load_npz(path: str):
    """Load pre-exported .npz and return arrays + train/val masks."""
    print(f"Loading {path}...")
    data = np.load(path, allow_pickle=True)

    embeddings = data["embeddings"]    # float32 [N, 768]
    is_real    = data["is_real"]       # bool    [N]
    cg_src     = data["cg_src"]        # int32   [N]
    cg_seg     = data["cg_seg"]        # int32   [N]
    vid_ids    = data["vid_ids"]       # object  [N]

    real_n = int(is_real.sum())
    fake_n = len(is_real) - real_n
    print(f"Segments: {len(embeddings):,}  |  "
          f"real: {real_n:,} ({100*real_n/len(embeddings):.1f}%)  |  "
          f"fake: {fake_n:,} ({100*fake_n/len(embeddings):.1f}%)  |  "
          f"RAM: {embeddings.nbytes/1e6:.0f} MB")

    # Stratified train/val split by video
    unique_vids = list(dict.fromkeys(vid_ids.tolist()))
    rng = random.Random(SEED)
    rng.shuffle(unique_vids)
    n_val       = max(1, int(len(unique_vids) * VAL_FRAC))
    val_vid_set = set(unique_vids[:n_val])

    train_mask = np.array([v not in val_vid_set for v in vid_ids])
    val_mask   = ~train_mask
    print(f"Train videos: {len(unique_vids)-n_val}  |  Val videos: {n_val}")
    print(f"Train segments: {int(train_mask.sum()):,}  |  Val segments: {int(val_mask.sum()):,}")

    cg_raw = list(zip(cg_src.tolist(), cg_seg.tolist()))
    return embeddings, is_real, cg_raw, vid_ids, train_mask, val_mask


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, is_real: np.ndarray,
                 cg_raw: List, min_group_size: int = 2):
        group_counts: Dict = defaultdict(int)
        for cg in cg_raw:
            group_counts[cg] += 1

        mask = np.array([group_counts[cg] >= min_group_size for cg in cg_raw])
        self.embeddings = embeddings[mask]
        self.is_real    = is_real[mask]
        cg_filtered     = [cg for cg, m in zip(cg_raw, mask) if m]

        unique_groups   = sorted(set(cg_filtered))
        group_to_id     = {g: i for i, g in enumerate(unique_groups)}
        self.cg_ids     = np.array([group_to_id[g] for g in cg_filtered], dtype=np.int64)

        removed = int((~mask).sum())
        if removed:
            print(f"  Filtered {removed:,} samples (groups <{min_group_size})")
        print(f"  Dataset: {len(self.embeddings):,} samples, {len(unique_groups):,} content groups")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "embedding":     torch.from_numpy(self.embeddings[idx]),
            "is_real":       torch.tensor(bool(self.is_real[idx]),  dtype=torch.bool),
            "content_group": torch.tensor(int(self.cg_ids[idx]),    dtype=torch.long),
        }


def collate(batch):
    return {
        "embeddings":     torch.stack([b["embedding"]     for b in batch]),
        "is_real":        torch.stack([b["is_real"]        for b in batch]),
        "content_groups": torch.stack([b["content_group"]  for b in batch]),
    }


def make_loaders(embeddings, is_real, cg_raw, train_mask, val_mask):
    train_ds = EmbeddingDataset(
        embeddings[train_mask], is_real[train_mask],
        [cg for cg, m in zip(cg_raw, train_mask) if m], min_group_size=2)
    val_ds   = EmbeddingDataset(
        embeddings[val_mask], is_real[val_mask],
        [cg for cg, m in zip(cg_raw, val_mask) if m], min_group_size=1)

    # WeightedRandomSampler: compensate for ~98% real / 2% fake imbalance
    train_labels  = (~train_ds.is_real).astype(int).tolist()
    class_counts  = [train_labels.count(0), train_labels.count(1)]
    class_weights = [1.0 / max(c, 1) for c in class_counts]
    sample_w      = [class_weights[l] for l in train_labels]
    sampler       = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate, num_workers=0)
    return train_loader, val_loader


# ── 4. Model ──────────────────────────────────────────────────────────────────

class DisentangledProjector(nn.Module):
    def __init__(self, input_dim: int = 768, output_dim: int = 128):
        super().__init__()
        self.f_auth = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, output_dim))
        self.f_id   = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, output_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, z):
        return F.normalize(self.f_auth(z), dim=-1), F.normalize(self.f_id(z), dim=-1)


# ── 5. Losses ─────────────────────────────────────────────────────────────────

def prototypical_contrastive_loss(z_id, content_groups, temperature=0.1):
    device = z_id.device
    unique_groups = torch.unique(content_groups)
    if len(unique_groups) < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    prototypes, proto_labels = [], []
    for gid in unique_groups:
        mask = content_groups == gid
        prototypes.append(z_id[mask].mean(dim=0))
        proto_labels.append(gid.item())

    prototypes = torch.stack(prototypes)
    distances  = -torch.cdist(z_id, prototypes, p=2)

    targets = torch.zeros(z_id.shape[0], dtype=torch.long, device=device)
    for i, gid in enumerate(proto_labels):
        targets[content_groups == gid] = i

    return F.cross_entropy(distances / temperature, targets)


def orthogonality_loss(z_id, z_auth, min_orth=0.001):
    sim = torch.matmul(z_id, z_auth.t())
    loss = sim.abs().sum() / (z_id.shape[0] ** 2)
    return torch.clamp(loss, min=min_orth)


class EqualWeightNormalizer:
    def __init__(self):
        self.initial_losses = {}
        self.step_count = 0

    def normalize_losses(self, L_proto, L_var, L_orth):
        self.step_count += 1
        if self.step_count == 1:
            self.initial_losses = {
                "proto": L_proto.item(),
                "var":   L_var.item(),
                "orth":  L_orth.item(),
            }
        eps = 1e-8
        return (
            L_proto / (self.initial_losses["proto"] + eps),
            L_var   / (self.initial_losses["var"]   + eps),
            L_orth  / (self.initial_losses["orth"]  + eps),
            {k: 1.0 / (v + eps) for k, v in self.initial_losses.items()},
        )


def variance_loss_with_repulsion(
    z_auth, is_real,
    min_variance=0.01, regularization_weight=0.1, repulsion_margin=0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    z_real = z_auth[is_real]
    z_fake = z_auth[~is_real]

    if z_real.shape[0] < 2:
        zero = torch.tensor(0.0, device=z_auth.device, requires_grad=True)
        return zero, zero

    mu_real = z_real.mean(dim=0, keepdim=True)
    attraction = ((z_real - mu_real) ** 2).sum(dim=1).mean()
    reg = (torch.clamp(min_variance - attraction, min=0.0) ** 2
           + 5.0 * torch.clamp(min_variance - attraction, min=0.0))
    attraction_loss = attraction + regularization_weight * reg

    if z_fake.shape[0] == 0:
        repulsion_loss = torch.tensor(0.0, device=z_auth.device, requires_grad=True)
    else:
        dist_fake = ((z_fake - mu_real.detach()) ** 2).sum(dim=1).sqrt()
        repulsion_loss = F.relu(repulsion_margin - dist_fake).mean()

    return attraction_loss, repulsion_loss


def compute_loss_baseline(z_id, z_auth, is_real, cg, normalizer):
    L_proto = prototypical_contrastive_loss(z_id, cg, TEMPERATURE)
    L_var, _ = variance_loss_with_repulsion(z_auth, is_real, MIN_VARIANCE, VAR_REG_WEIGHT)
    L_orth = orthogonality_loss(z_id.detach(), z_auth, MIN_ORTH)

    Lp, Lv, Lo, _ = normalizer.normalize_losses(L_proto, L_var, L_orth)
    total = Lp + Lv + Lo
    return total, {"total": total.item(), "proto": L_proto.item(),
                   "var": L_var.item(), "orth": L_orth.item(), "repel": 0.0}


def compute_loss_fix1a(z_id, z_auth, is_real, cg, normalizer, repel_init):
    L_proto = prototypical_contrastive_loss(z_id, cg, TEMPERATURE)
    L_var, L_repel = variance_loss_with_repulsion(
        z_auth, is_real, MIN_VARIANCE, VAR_REG_WEIGHT, REPULSION_MARGIN)
    L_orth = orthogonality_loss(z_id.detach(), z_auth, MIN_ORTH)

    Lp, Lv, Lo, _ = normalizer.normalize_losses(L_proto, L_var, L_orth)

    if "val" not in repel_init:
        repel_init["val"] = max(L_repel.item(), 1e-6)
    L_repel_norm = L_repel / (repel_init["val"] + 1e-8)

    total = Lp + Lv + Lo + LAMBDA_REPEL * L_repel_norm
    return total, {"total": total.item(), "proto": L_proto.item(),
                   "var": L_var.item(), "orth": L_orth.item(), "repel": L_repel.item()}


# ── 6. Metrics ────────────────────────────────────────────────────────────────

def compute_clustering_metrics(embs, labels):
    labels = labels.astype(int)
    km = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
    cl = km.fit_predict(embs)
    return {
        "ami":                adjusted_mutual_info_score(labels, cl),
        "ari":                adjusted_rand_score(labels, cl),
        "silhouette_gt":      float(silhouette_score(embs, labels,    metric="cosine")),
        "silhouette_clusters": float(silhouette_score(embs, cl,       metric="cosine")),
    }


def compute_distribution_metrics(embs, labels, n_bins=50):
    labels = labels.astype(int)
    real_embs = embs[labels == 0]
    fake_embs = embs[labels == 1]
    if len(real_embs) == 0 or len(fake_embs) == 0:
        return {"kl_divergence": 0.0, "js_distance": 0.0, "wasserstein_distance": 0.0}

    centroid     = real_embs.mean(axis=0)
    real_dists   = np.linalg.norm(real_embs - centroid, axis=1)
    fake_dists   = np.linalg.norm(fake_embs - centroid, axis=1)
    all_d        = np.concatenate([real_dists, fake_dists])
    lo, hi       = all_d.min(), all_d.max()
    if hi - lo < 1e-10:
        return {"kl_divergence": 0.0, "js_distance": 0.0, "wasserstein_distance": 0.0}

    bins = np.linspace(lo, hi, n_bins + 1)
    rh, _ = np.histogram(real_dists, bins=bins)
    fh, _ = np.histogram(fake_dists, bins=bins)
    rp = (rh.astype(float) + 1e-10); rp /= rp.sum()
    fp = (fh.astype(float) + 1e-10); fp /= fp.sum()
    return {
        "kl_divergence":      float(np.sum(rp * np.log(rp / fp))),
        "js_distance":        float(jensenshannon(rp, fp)),
        "wasserstein_distance": float(wasserstein_distance(real_dists, fake_dists)),
    }


def compute_separation_metrics(embs, labels):
    labels = labels.astype(int)
    real_embs = embs[labels == 0]
    fake_embs = embs[labels == 1]
    if len(real_embs) == 0 or len(fake_embs) == 0:
        return {"separation_gap": 0.0, "mean_distance_real": 0.0,
                "mean_distance_fake": 0.0}

    centroid = real_embs.mean(axis=0)
    cn = centroid / (np.linalg.norm(centroid) + 1e-10)

    def norm_rows(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(n < 1e-10, 1.0, n)

    sim_rr = norm_rows(real_embs) @ cn
    sim_fr = norm_rows(fake_embs) @ cn
    return {
        "separation_gap":     float(sim_rr.mean() - sim_fr.mean()),
        "mean_distance_real": float(np.linalg.norm(real_embs - centroid, axis=1).mean()),
        "mean_distance_fake": float(np.linalg.norm(fake_embs - centroid, axis=1).mean()),
    }


def compute_all_metrics(embs, labels):
    m = {}
    m.update(compute_clustering_metrics(embs, labels))
    m.update(compute_distribution_metrics(embs, labels))
    m.update(compute_separation_metrics(embs, labels))
    return m


# ── 7. Evaluation helper ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader, device, use_auth=True):
    model.eval()
    all_emb, all_lbl = [], []
    n = 0
    for batch in loader:
        emb = batch["embeddings"].to(device)
        lbl = (~batch["is_real"]).long()
        if use_auth:
            z_auth, _ = model(emb)
            out = z_auth
        else:
            out = emb
        all_emb.append(out.cpu())
        all_lbl.append(lbl.cpu())
        n += emb.shape[0]
        if n >= MAX_EVAL_SAMPLES:
            break
    embs   = torch.cat(all_emb).numpy()[:MAX_EVAL_SAMPLES]
    labels = torch.cat(all_lbl).numpy()[:MAX_EVAL_SAMPLES]
    return embs, labels


# ── 8. Plotting ───────────────────────────────────────────────────────────────

def _scatter(ax, embs_2d, labels, title, max_pts=2000):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(embs_2d), size=min(max_pts, len(embs_2d)), replace=False)
    e, l = embs_2d[idx], labels[idx]
    colors = np.where(l == 0, "#2196F3", "#F44336")
    ax.scatter(e[:, 0], e[:, 1], c=colors, alpha=0.35, s=6, linewidths=0)
    ax.scatter([], [], c="#2196F3", s=20, label="real")
    ax.scatter([], [], c="#F44336", s=20, label="fake")
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def plot_embeddings(raw_embs, raw_lbl, bl_after, bl_lbl, f1_after, f1_lbl, save_path):
    sets = [raw_embs, bl_after, f1_after]
    lbls = [raw_lbl, bl_lbl, f1_lbl]
    titles_pca = ["Raw HuBERT input", "Baseline z_auth (final)", "Fix1a z_auth (final)"]

    print("  PCA...")
    pcas = [PCA(n_components=2, random_state=0).fit_transform(e) for e in sets]

    print("  t-SNE (may take ~1 min)...")
    n_tsne = 1500
    rng = np.random.default_rng(0)
    tsneds, tidxs = [], []
    for e in sets:
        idx = rng.choice(len(e), size=min(n_tsne, len(e)), replace=False)
        ts  = TSNE(n_components=2, perplexity=30, random_state=0, n_iter=500).fit_transform(e[idx])
        tsneds.append(ts); tidxs.append(idx)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        f"AVDeepFake1M 20% — Baseline vs Fix1a (margin={REPULSION_MARGIN})\n"
        f"Blue=real  Red=fake  |  {MAX_EVAL_SAMPLES} samples max", fontsize=10)

    for col, (p, l, t) in enumerate(zip(pcas, lbls, titles_pca)):
        _scatter(axes[0, col], p, l, f"PCA: {t}")
    for col, (ts, idx, l, t) in enumerate(zip(tsneds, tidxs, lbls, titles_pca)):
        _scatter(axes[1, col], ts, l[idx], f"t-SNE: {t}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_training_curves(hist_baseline, hist_fix1a, save_path):
    epochs = range(1, len(hist_baseline) + 1)

    def get(hist, key):
        return [e["losses"][key] for e in hist]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Training curves — Baseline vs Fix1a", fontsize=11)

    for i, (key, label) in enumerate([
        ("total", "Total loss"), ("proto", "Proto loss"),
        ("var",   "Var (attraction) loss"), ("orth",  "Orth loss"),
        ("repel", "Repulsion loss"),
    ]):
        ax = axes[i // 3, i % 3]
        ax.plot(epochs, get(hist_baseline, key), label="baseline", color="steelblue")
        ax.plot(epochs, get(hist_fix1a,    key), label="fix1a",    color="tomato")
        ax.set_title(label, fontsize=9); ax.legend(fontsize=7); ax.set_xlabel("epoch", fontsize=8)

    ax = axes[1, 2]
    ax.plot(epochs, [e["metrics"]["silhouette_gt"] for e in hist_baseline], label="baseline", color="steelblue")
    ax.plot(epochs, [e["metrics"]["silhouette_gt"] for e in hist_fix1a],    label="fix1a",    color="tomato")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Silhouette (z_auth, GT labels)", fontsize=9)
    ax.legend(fontsize=7); ax.set_xlabel("epoch", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── 9. Training loop ──────────────────────────────────────────────────────────

def run_experiment(name, train_loader, val_loader, input_dim, device, use_repulsion):
    model      = DisentangledProjector(input_dim, 128).to(device)
    optimizer  = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    normalizer = EqualWeightNormalizer()
    repel_init = {}
    scheduler  = LambdaLR(optimizer, lambda s: min(1.0, s / WARMUP_STEPS))
    history    = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        ep_losses = defaultdict(float)
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"{name} ep{epoch}/{NUM_EPOCHS}", leave=False):
            emb  = batch["embeddings"].to(device)
            real = batch["is_real"].to(device)
            cg   = batch["content_groups"].to(device)
            z_auth, z_id = model(emb)

            if use_repulsion:
                total, ld = compute_loss_fix1a(z_id, z_auth, real, cg, normalizer, repel_init)
            else:
                total, ld = compute_loss_baseline(z_id, z_auth, real, cg, normalizer)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            for k, v in ld.items():
                ep_losses[k] += v
            n_batches += 1

        avg = {k: v / n_batches for k, v in ep_losses.items()}
        embs, lbls = extract_embeddings(model, val_loader, device)
        metrics    = compute_all_metrics(embs, lbls)
        history.append({"epoch": epoch, "losses": avg, "metrics": metrics})

        print(f"[{name}] ep{epoch:02d}  "
              f"total={avg['total']:.4f}  proto={avg['proto']:.4f}  "
              f"var={avg['var']:.4f}  orth={avg['orth']:.4f}  "
              f"repel={avg['repel']:.4f}  "
              f"sil_gt={metrics['silhouette_gt']:.4f}  "
              f"sil_cl={metrics['silhouette_clusters']:.4f}  "
              f"wass={metrics['wasserstein_distance']:.4f}")

    return model, history


# ── 10. Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = (torch.device("cuda")  if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}\n")

    # Load data
    embeddings, is_real, cg_raw, vid_ids, train_mask, val_mask = load_npz(NPZ_PATH)
    train_loader, val_loader = make_loaders(embeddings, is_real, cg_raw, train_mask, val_mask)

    input_dim = embeddings.shape[-1]
    print(f"\nInput dim: {input_dim}  |  Batch size: {BATCH_SIZE}\n")

    # Capture raw embeddings for plotting
    print("Capturing raw embeddings...")
    raw_model = DisentangledProjector(input_dim, 128).to(device)
    raw_embs, raw_lbls = extract_embeddings(raw_model, val_loader, device, use_auth=False)
    print(f"  Raw: {raw_embs.shape}  |  fake ratio: {raw_lbls.mean():.3f}\n")

    # Run experiments
    print("=" * 60)
    print("EXPERIMENT A: Baseline (no repulsion)")
    print("=" * 60)
    bl_model, bl_hist = run_experiment(
        "baseline", train_loader, val_loader, input_dim, device, use_repulsion=False)

    print("\n" + "=" * 60)
    print(f"EXPERIMENT B: Fix 1a (repulsion margin={REPULSION_MARGIN})")
    print("=" * 60)
    f1_model, f1_hist = run_experiment(
        "fix1a", train_loader, val_loader, input_dim, device, use_repulsion=True)

    # Final embeddings for plots
    print("\nExtracting final embeddings...")
    bl_after, bl_lbl = extract_embeddings(bl_model, val_loader, device)
    f1_after, f1_lbl = extract_embeddings(f1_model, val_loader, device)

    # Plots
    print("\nGenerating plots...")
    plot_embeddings(raw_embs, raw_lbls, bl_after, bl_lbl, f1_after, f1_lbl,
                    os.path.join(SAVE_DIR, "embeddings_before_after.png"))
    plot_training_curves(bl_hist, f1_hist,
                         os.path.join(SAVE_DIR, "training_curves.png"))

    # Summary
    def best_sil(hist): return max(e["metrics"]["silhouette_gt"] for e in hist)

    raw_m = compute_all_metrics(raw_embs, raw_lbls)
    bl_m  = bl_hist[-1]["metrics"]
    f1_m  = f1_hist[-1]["metrics"]

    print("\n" + "=" * 80)
    print("FINAL METRICS SUMMARY")
    print("=" * 80)
    for label, m in [("Raw HuBERT input", raw_m), ("Baseline final", bl_m), ("Fix1a final", f1_m)]:
        print(f"{label:25s}  sil_gt={m['silhouette_gt']:+.4f}  "
              f"sil_cl={m['silhouette_clusters']:.4f}  "
              f"wass={m['wasserstein_distance']:.4f}  "
              f"sep_gap={m['separation_gap']:+.4f}")
    print(f"{'Best sil_gt':25s}  baseline={best_sil(bl_hist):+.4f}  fix1a={best_sil(f1_hist):+.4f}")
    print("=" * 80)

    # Save results
    results = {
        "config": {
            "repulsion_margin": REPULSION_MARGIN, "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS, "lr": LR, "lambda_repel": LAMBDA_REPEL, "seed": SEED,
        },
        "raw_metrics":      {k: float(v) for k, v in raw_m.items()},
        "baseline_history": bl_hist,
        "fix1a_history":    f1_hist,
    }
    out_path = os.path.join(SAVE_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out_path}")

    torch.save(bl_model.state_dict(), os.path.join(SAVE_DIR, "baseline_model.pt"))
    torch.save(f1_model.state_dict(), os.path.join(SAVE_DIR, "fix1a_model.pt"))
    print("Models saved.")


if __name__ == "__main__":
    main()
