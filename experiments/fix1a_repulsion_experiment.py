"""
Fix 1a: Hard-margin fake repulsion experiment.

Compares:
  - Baseline: equal-weight normalizer, no repulsion (mirrors disentangled_equal_weights run)
  - Fix1a:    same + relu(0.5 - dist(z_fake, mu_real.detach())).mean() repulsion term

Dataset: AVDeepFake1M only, 20% stratified subset (by video).

NOTE: AVDeepFake1M has a strongly imbalanced segment split: ~98.9% real / 1.1% fake
at the segment level (partial audio fakes within otherwise real videos). This is the
natural split we preserve per the experiment spec. Expect small fake counts per batch
(~1-2 at batch_size=128); repulsion will fire sparsely but signal should accumulate.

Run:
    /Users/ricardocarrillo/itau-group4/venv/bin/python3 -m experiments.fix1a_repulsion_experiment
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.disentangled.model import DisentangledProjector
from training.disentangled.losses import (
    prototypical_contrastive_loss,
    orthogonality_loss,
    EqualWeightNormalizer,
)
from training.disentangled.metrics import (
    compute_clustering_metrics,
    compute_distribution_metrics,
    compute_separation_metrics,
)

# ── Config ────────────────────────────────────────────────────────────────────

HDF5_PATH      = "exports/deepfake_embeddings.h5"
ENCODER        = "hubert"
SUBSET_FRAC    = 0.20          # 20% of AVDeepFake1M videos
REPULSION_MARGIN = 0.50        # Fix 1a margin
VAL_FRAC       = 0.20
BATCH_SIZE     = 128
NUM_EPOCHS     = 20
LR             = 1e-4
WEIGHT_DECAY   = 1e-5
WARMUP_STEPS   = 500
TEMPERATURE    = 0.1
MIN_VARIANCE   = 0.01
VAR_REG_WEIGHT = 0.1
MIN_ORTH       = 0.001
LAMBDA_VAR     = 1.0           # raw weight before EqualWeightNormalizer scales
LAMBDA_ORTH    = 1.0
LAMBDA_REPEL   = 1.0           # raw weight for repulsion; normalizer handles scaling
SEED           = 42
SAVE_DIR       = "results/disentangled/fix1a_repulsion"
MAX_EVAL_SAMPLES = 5000        # cap for metric computation speed

# ── Reproducibility ───────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_avdeepfake_subset(hdf5_path: str, encoder: str, subset_frac: float, seed: int = 42):
    """
    Load a stratified 20% subset of AVDeepFake1M videos.
    All embeddings are read from HDF5 in one pass and stored in RAM as numpy arrays
    so __getitem__ never touches disk again.

    Returns (embeddings np.ndarray [N, D], is_real [N], content_group_id [N],
             video_ids [N], real_n, fake_n)
    """
    rng = random.Random(seed)

    with h5py.File(hdf5_path, 'r') as f:
        all_vids = list(f['videos'].keys())
        av_vids  = [v for v in all_vids if f['videos'][v].attrs.get('dataset', '') == 'avdeepfake1m']

    n_subset = max(1, int(len(av_vids) * subset_frac))
    selected = sorted(rng.sample(av_vids, n_subset))

    all_embs, all_real, all_cg, all_vid = [], [], [], []
    source_counter = 0

    print(f"Reading {len(selected)} videos into RAM...")
    with h5py.File(hdf5_path, 'r') as f:
        for video_id in tqdm(selected, desc="Loading HDF5", leave=False):
            v = f['videos'][video_id]
            if f'embeddings/{encoder}' not in v:
                continue
            if 'labels' not in v or 'audio' not in v['labels']:
                continue

            embs_vid   = v[f'embeddings/{encoder}'][:]   # (num_augs, num_segs, D)
            audio_lbl  = v['labels/audio'][:]             # (num_augs, num_segs)
            num_augs, num_segs, D = embs_vid.shape

            for aug_idx in range(num_augs):
                for seg_idx in range(num_segs):
                    all_embs.append(embs_vid[aug_idx, seg_idx])
                    all_real.append(audio_lbl[aug_idx, seg_idx] == 0.0)
                    all_cg.append((source_counter, seg_idx))
                    all_vid.append(video_id)

            source_counter += 1

    embeddings    = np.stack(all_embs).astype(np.float32)   # [N, D]
    is_real_arr   = np.array(all_real, dtype=bool)           # [N]
    cg_raw        = all_cg                                   # list of (src, seg) tuples
    vid_ids       = all_vid

    real_n = int(is_real_arr.sum())
    fake_n = len(is_real_arr) - real_n
    mem_mb = embeddings.nbytes / 1e6
    print(f"Loaded {len(selected)} AVDeepFake1M videos ({subset_frac*100:.0f}% of {len(av_vids)})")
    print(f"Segments: {len(embeddings):,}  |  real: {real_n:,} ({100*real_n/len(embeddings):.1f}%)  "
          f"|  fake: {fake_n:,} ({100*fake_n/len(embeddings):.1f}%)  |  RAM: {mem_mb:.0f} MB")

    return embeddings, is_real_arr, cg_raw, vid_ids, real_n, fake_n


class AVDeepFakeSubsetDataset(Dataset):
    """
    In-memory dataset — all embeddings pre-loaded as numpy arrays.
    __getitem__ is a pure array index: no I/O, no file opens.
    """

    def __init__(self, embeddings: np.ndarray, is_real: np.ndarray,
                 cg_raw: List, min_group_size: int = 2):

        # Filter to content groups with >= min_group_size samples
        group_counts: Dict = defaultdict(int)
        for cg in cg_raw:
            group_counts[cg] += 1

        mask = np.array([group_counts[cg] >= min_group_size for cg in cg_raw])
        self.embeddings = embeddings[mask]
        self.is_real    = is_real[mask]
        cg_filtered     = [cg for cg, m in zip(cg_raw, mask) if m]

        # Map content groups to compact integer IDs
        unique_groups = sorted(set(cg_filtered))
        group_to_id   = {g: i for i, g in enumerate(unique_groups)}
        self.cg_ids   = np.array([group_to_id[g] for g in cg_filtered], dtype=np.int64)

        removed = int((~mask).sum())
        if removed:
            print(f"  Filtered {removed:,} samples from groups with <{min_group_size} members")
        print(f"  Dataset ready: {len(self.embeddings):,} samples, {len(unique_groups):,} content groups")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> Dict:
        return {
            'embedding':     torch.from_numpy(self.embeddings[idx]),
            'is_real':       torch.tensor(bool(self.is_real[idx]), dtype=torch.bool),
            'content_group': torch.tensor(int(self.cg_ids[idx]),   dtype=torch.long),
        }


def make_collate(batch):
    return {
        'embeddings':     torch.stack([b['embedding'] for b in batch]),
        'is_real':        torch.stack([b['is_real'] for b in batch]),
        'content_groups': torch.stack([b['content_group'] for b in batch]),
    }


# ── Modified loss ─────────────────────────────────────────────────────────────

def variance_loss_with_repulsion(
    z_auth: torch.Tensor,
    is_real: torch.Tensor,
    min_variance: float = 0.01,
    regularization_weight: float = 0.1,
    repulsion_margin: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (attraction_loss, repulsion_loss) separately so the caller can
    weight them independently before summing.

    Attraction: pull real samples toward mu_real  (same as original variance_loss)
    Repulsion:  push fake samples at least `repulsion_margin` from mu_real
                using relu(margin - ||z_fake - mu_real.detach()||_2).mean()

    mu_real is DETACHED in the repulsion term so the gradient from repulsion
    only moves fake embeddings, not the real centroid.
    """
    z_real = z_auth[is_real]
    z_fake = z_auth[~is_real]

    if z_real.shape[0] < 2:
        zero = torch.tensor(0.0, device=z_auth.device, requires_grad=True)
        return zero, zero

    mu_real = z_real.mean(dim=0, keepdim=True)              # [1, D]

    # Attraction
    attraction = ((z_real - mu_real) ** 2).sum(dim=1).mean()
    reg_quad   = torch.clamp(min_variance - attraction, min=0.0) ** 2
    reg_lin    = torch.clamp(min_variance - attraction, min=0.0)
    attraction_loss = attraction + regularization_weight * (reg_quad + 5.0 * reg_lin)

    # Repulsion — detach centroid so gradient flows only to z_fake
    if z_fake.shape[0] == 0:
        repulsion_loss = torch.tensor(0.0, device=z_auth.device, requires_grad=True)
    else:
        mu_real_sg = mu_real.detach()                        # stop-gradient on centroid
        dist_fake  = ((z_fake - mu_real_sg) ** 2).sum(dim=1).sqrt()
        repulsion_loss = F.relu(repulsion_margin - dist_fake).mean()

    return attraction_loss, repulsion_loss


def compute_loss_baseline(z_id, z_auth, is_real, content_groups, normalizer,
                          temperature, min_variance, var_reg_weight, min_orth):
    """Original three-loss combination (no repulsion)."""
    L_proto = prototypical_contrastive_loss(z_id, content_groups, temperature=temperature)
    L_var_attract, _ = variance_loss_with_repulsion(z_auth, is_real, min_variance, var_reg_weight)
    L_orth   = orthogonality_loss(z_id.detach(), z_auth, min_orth=min_orth)  # stop-grad on z_id

    L_p, L_v, L_o, _ = normalizer.normalize_losses(L_proto, L_var_attract, L_orth)
    total = L_p + L_v + L_o

    return total, {
        'total': total.item(), 'proto': L_proto.item(),
        'var': L_var_attract.item(), 'orth': L_orth.item(), 'repel': 0.0,
    }


def compute_loss_fix1a(z_id, z_auth, is_real, content_groups, normalizer, repel_init,
                       temperature, min_variance, var_reg_weight, min_orth, margin, lambda_repel):
    """Three-loss + repulsion. EqualWeightNormalizer handles proto/var/orth; repulsion
    is divided by its initial value so it also starts at scale ~1.0."""
    L_proto = prototypical_contrastive_loss(z_id, content_groups, temperature=temperature)
    L_var_attract, L_repel = variance_loss_with_repulsion(
        z_auth, is_real, min_variance, var_reg_weight, repulsion_margin=margin)
    L_orth = orthogonality_loss(z_id.detach(), z_auth, min_orth=min_orth)

    L_p, L_v, L_o, _ = normalizer.normalize_losses(L_proto, L_var_attract, L_orth)

    # Normalize repulsion by initial value (set on first call via repel_init dict)
    if 'val' not in repel_init:
        repel_init['val'] = max(L_repel.item(), 1e-6)
    L_repel_norm = L_repel / (repel_init['val'] + 1e-8)

    total = L_p + L_v + L_o + lambda_repel * L_repel_norm

    return total, {
        'total': total.item(), 'proto': L_proto.item(),
        'var': L_var_attract.item(), 'orth': L_orth.item(), 'repel': L_repel.item(),
    }


# ── Evaluation helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, loader, device, use_auth=True, max_samples=MAX_EVAL_SAMPLES):
    """Extract z_auth (or raw input) embeddings + labels."""
    model.eval()
    all_emb, all_lbl = [], []
    n = 0
    for batch in loader:
        emb  = batch['embeddings'].to(device)
        lbl  = (~batch['is_real']).long()   # 0=real, 1=fake

        if use_auth:
            z_auth, _ = model(emb)
            out = z_auth
        else:
            out = emb

        all_emb.append(out.cpu())
        all_lbl.append(lbl.cpu())
        n += emb.shape[0]
        if n >= max_samples:
            break

    embs   = torch.cat(all_emb).numpy()[:max_samples]
    labels = torch.cat(all_lbl).numpy()[:max_samples]
    return embs, labels


def compute_all_metrics(embs, labels):
    m = {}
    m.update(compute_clustering_metrics(embs, labels))
    m.update(compute_distribution_metrics(embs, labels))
    m.update(compute_separation_metrics(embs, labels))
    return m


# ── Plotting ──────────────────────────────────────────────────────────────────

def _scatter(ax, embs_2d, labels, title, max_pts=2000):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(embs_2d), size=min(max_pts, len(embs_2d)), replace=False)
    e, l = embs_2d[idx], labels[idx]
    colors = np.where(l == 0, '#2196F3', '#F44336')   # blue=real, red=fake
    ax.scatter(e[:, 0], e[:, 1], c=colors, alpha=0.35, s=6, linewidths=0)
    # Legend proxies
    ax.scatter([], [], c='#2196F3', s=20, label='real')
    ax.scatter([], [], c='#F44336', s=20, label='fake')
    ax.legend(fontsize=7, markerscale=2)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def plot_embeddings(
    raw_embs, raw_labels,
    baseline_before, baseline_before_lbl,
    baseline_after,  baseline_after_lbl,
    fix1a_before,    fix1a_before_lbl,
    fix1a_after,     fix1a_after_lbl,
    save_path: str,
):
    """
    2×4 grid:
      Row 0: PCA  — raw input | baseline before | baseline after | fix1a after
      Row 1: t-SNE — same columns
    """
    n_tsne = 1500

    def fit_pca(embs):
        return PCA(n_components=2, random_state=0).fit_transform(embs)

    def fit_tsne(embs, n=n_tsne):
        idx = np.random.default_rng(0).choice(len(embs), size=min(n, len(embs)), replace=False)
        return TSNE(n_components=2, perplexity=30, random_state=0, n_iter=500).fit_transform(embs[idx]), idx

    print("  Computing PCA projections...")
    sets = [raw_embs, baseline_before, baseline_after, fix1a_after]
    pcas = [fit_pca(e) for e in sets]
    lbl_list = [raw_labels, baseline_before_lbl, baseline_after_lbl, fix1a_after_lbl]
    titles_pca = [
        "Raw HuBERT input",
        "Baseline z_auth (init)",
        "Baseline z_auth (final)",
        "Fix1a z_auth (final)",
    ]

    print("  Computing t-SNE projections (this may take ~1 min)...")
    tsneds, idxs = zip(*[fit_tsne(e) for e in sets])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"AVDeepFake1M 20% subset — Baseline vs Fix1a (hard-margin repulsion={REPULSION_MARGIN})\n"
        f"Blue=real  Red=fake  |  {MAX_EVAL_SAMPLES} samples max",
        fontsize=10,
    )

    for col, (pca2d, lbl, title) in enumerate(zip(pcas, lbl_list, titles_pca)):
        _scatter(axes[0, col], pca2d, lbl, f"PCA: {title}")

    titles_tsne = [t.replace("PCA", "t-SNE") for t in titles_pca]
    for col, (ts2d, idx, lbl, title) in enumerate(zip(tsneds, idxs, lbl_list, titles_tsne)):
        _scatter(axes[1, col], ts2d, lbl[idx], f"t-SNE: {title}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved embedding plot → {save_path}")


def plot_training_curves(hist_baseline, hist_fix1a, save_path):
    epochs = range(1, len(hist_baseline) + 1)

    def get(hist, key):
        return [e['losses'][key] for e in hist]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Training curves — Baseline vs Fix1a", fontsize=11)

    metrics_to_plot = [
        ('total', 'Total loss'),
        ('proto', 'Proto loss'),
        ('var',   'Var (attraction) loss'),
        ('orth',  'Orth loss'),
        ('repel', 'Repulsion loss'),
    ]

    sil_baseline = [e['metrics']['silhouette_gt'] for e in hist_baseline]
    sil_fix1a    = [e['metrics']['silhouette_gt'] for e in hist_fix1a]

    for i, (key, label) in enumerate(metrics_to_plot):
        ax = axes[i // 3, i % 3]
        ax.plot(epochs, get(hist_baseline, key), label='baseline', color='steelblue')
        ax.plot(epochs, get(hist_fix1a,    key), label='fix1a',    color='tomato')
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel('epoch', fontsize=8)

    ax = axes[1, 2]
    ax.plot(epochs, sil_baseline, label='baseline', color='steelblue')
    ax.plot(epochs, sil_fix1a,    label='fix1a',    color='tomato')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title('Silhouette (z_auth, GT labels)', fontsize=9)
    ax.legend(fontsize=7)
    ax.set_xlabel('epoch', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curve plot → {save_path}")


# ── Training loop ─────────────────────────────────────────────────────────────

def run_one_experiment(
    name: str,
    train_loader, val_loader,
    input_dim: int,
    device: torch.device,
    use_repulsion: bool,
) -> Tuple[DisentangledProjector, List[Dict]]:

    model     = DisentangledProjector(input_dim=input_dim, output_dim=128).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    normalizer = EqualWeightNormalizer()
    repel_init = {}

    def lr_lambda(step):
        return min(1.0, step / WARMUP_STEPS)
    scheduler = LambdaLR(optimizer, lr_lambda)

    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        ep_losses = defaultdict(float)
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"{name} ep{epoch}/{NUM_EPOCHS}", leave=False):
            emb  = batch['embeddings'].to(device)
            real = batch['is_real'].to(device)
            cg   = batch['content_groups'].to(device)

            z_auth, z_id = model(emb)

            if use_repulsion:
                total, ld = compute_loss_fix1a(
                    z_id, z_auth, real, cg, normalizer, repel_init,
                    TEMPERATURE, MIN_VARIANCE, VAR_REG_WEIGHT, MIN_ORTH,
                    REPULSION_MARGIN, LAMBDA_REPEL,
                )
            else:
                total, ld = compute_loss_baseline(
                    z_id, z_auth, real, cg, normalizer,
                    TEMPERATURE, MIN_VARIANCE, VAR_REG_WEIGHT, MIN_ORTH,
                )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            for k, v in ld.items():
                ep_losses[k] += v
            n_batches += 1

        avg_losses = {k: v / n_batches for k, v in ep_losses.items()}

        # ── Eval ───────────────────────────────────────────────
        embs, lbls = extract_embeddings(model, val_loader, device, use_auth=True)
        metrics    = compute_all_metrics(embs, lbls)

        history.append({'epoch': epoch, 'losses': avg_losses, 'metrics': metrics})

        print(
            f"[{name}] ep{epoch:02d}  "
            f"total={avg_losses['total']:.4f}  proto={avg_losses['proto']:.4f}  "
            f"var={avg_losses['var']:.4f}  orth={avg_losses['orth']:.4f}  "
            f"repel={avg_losses['repel']:.4f}  "
            f"sil_gt={metrics['silhouette_gt']:.4f}  "
            f"sil_clust={metrics['silhouette_clusters']:.4f}  "
            f"wasserstein={metrics['wasserstein_distance']:.4f}"
        )

    return model, history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps')  if torch.backends.mps.is_available()
              else torch.device('cpu'))
    print(f"Device: {device}\n")

    # ── Load data ──────────────────────────────────────────────
    print("=" * 60)
    print("Loading AVDeepFake1M 20% subset")
    print("=" * 60)
    embeddings, is_real_arr, cg_raw, vid_ids_all, real_n, fake_n = load_avdeepfake_subset(
        HDF5_PATH, ENCODER, SUBSET_FRAC, SEED)

    # Stratified train/val split by video (80/20)
    unique_vids = list(dict.fromkeys(vid_ids_all))
    rng         = random.Random(SEED)
    rng.shuffle(unique_vids)
    n_val       = max(1, int(len(unique_vids) * VAL_FRAC))
    val_vid_set = set(unique_vids[:n_val])

    train_mask = np.array([v not in val_vid_set for v in vid_ids_all])
    val_mask   = ~train_mask

    print(f"\nTrain videos: {len(unique_vids)-n_val}  |  Val videos: {n_val}")
    print(f"Train segments: {int(train_mask.sum()):,}  |  Val segments: {int(val_mask.sum()):,}\n")

    train_ds = AVDeepFakeSubsetDataset(
        embeddings[train_mask], is_real_arr[train_mask],
        [cg for cg, m in zip(cg_raw, train_mask) if m], min_group_size=2)
    val_ds   = AVDeepFakeSubsetDataset(
        embeddings[val_mask], is_real_arr[val_mask],
        [cg for cg, m in zip(cg_raw, val_mask) if m], min_group_size=1)

    # WeightedRandomSampler: balance batches despite 98.9/1.1 real/fake split
    train_labels  = (~train_ds.is_real).astype(int).tolist()  # 0=real, 1=fake
    class_counts  = [train_labels.count(0), train_labels.count(1)]
    class_weights = [1.0 / max(c, 1) for c in class_counts]
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              collate_fn=make_collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=make_collate, num_workers=0)

    # Determine input_dim from first batch
    sample_batch  = next(iter(val_loader))
    input_dim = sample_batch['embeddings'].shape[-1]
    print(f"Input dim: {input_dim}  |  Batch size: {BATCH_SIZE}\n")

    # ── Capture raw input embeddings for plotting ───────────────
    print("Capturing raw input embeddings for baseline plot...")
    raw_model = DisentangledProjector(input_dim=input_dim, output_dim=128).to(device)
    raw_embs, raw_lbls = extract_embeddings(raw_model, val_loader, device, use_auth=False)
    print(f"  Raw embeddings: {raw_embs.shape}  |  fake ratio: {raw_lbls.mean():.3f}\n")

    # ── Run baseline ───────────────────────────────────────────
    print("=" * 60)
    print("EXPERIMENT A: Baseline (no repulsion)")
    print("=" * 60)
    baseline_model, baseline_hist = run_one_experiment(
        "baseline", train_loader, val_loader, input_dim, device, use_repulsion=False)

    # ── Run Fix 1a ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"EXPERIMENT B: Fix 1a (hard-margin repulsion, margin={REPULSION_MARGIN})")
    print("=" * 60)
    fix1a_model, fix1a_hist = run_one_experiment(
        "fix1a", train_loader, val_loader, input_dim, device, use_repulsion=True)

    # ── Collect final embeddings for plotting ──────────────────
    print("\nExtracting final embeddings for plots...")
    bl_before_embs, bl_before_lbls = raw_embs, raw_lbls   # same as raw
    bl_after_embs,  bl_after_lbls  = extract_embeddings(baseline_model, val_loader, device, use_auth=True)
    f1_before_embs, f1_before_lbls = raw_embs, raw_lbls
    f1_after_embs,  f1_after_lbls  = extract_embeddings(fix1a_model,    val_loader, device, use_auth=True)

    # ── Plots ──────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_embeddings(
        raw_embs, raw_lbls,
        bl_before_embs, bl_before_lbls,
        bl_after_embs,  bl_after_lbls,
        f1_before_embs, f1_before_lbls,
        f1_after_embs,  f1_after_lbls,
        save_path=os.path.join(SAVE_DIR, "embeddings_before_after.png"),
    )
    plot_training_curves(
        baseline_hist, fix1a_hist,
        save_path=os.path.join(SAVE_DIR, "training_curves.png"),
    )

    # ── Metrics summary ────────────────────────────────────────
    def last_m(hist):  return hist[-1]['metrics']
    def best_sil(hist): return max(e['metrics']['silhouette_gt'] for e in hist)

    raw_metrics = compute_all_metrics(raw_embs, raw_lbls)
    bl_final    = last_m(baseline_hist)
    f1_final    = last_m(fix1a_hist)

    def row(label, m):
        return (
            f"{label:30s}  "
            f"sil_gt={m['silhouette_gt']:+.4f}  "
            f"sil_cl={m['silhouette_clusters']:.4f}  "
            f"ami={m['ami']:.4f}  "
            f"wasserstein={m['wasserstein_distance']:.4f}  "
            f"sep_gap={m['separation_gap']:+.4f}  "
            f"dist_fake={m['mean_distance_fake']:.4f}  "
            f"dist_real={m['mean_distance_real']:.4f}"
        )

    print("\n" + "=" * 80)
    print("FINAL METRICS SUMMARY")
    print("=" * 80)
    print(row("Raw HuBERT input",          raw_metrics))
    print(row("Baseline (final epoch)",    bl_final))
    print(row("Fix1a   (final epoch)",     f1_final))
    print(f"\n{'Best silhouette_gt':30s}  baseline={best_sil(baseline_hist):+.4f}  fix1a={best_sil(fix1a_hist):+.4f}")
    print("=" * 80)

    # ── Save results ───────────────────────────────────────────
    results = {
        'config': {
            'encoder': ENCODER, 'subset_frac': SUBSET_FRAC, 'repulsion_margin': REPULSION_MARGIN,
            'batch_size': BATCH_SIZE, 'num_epochs': NUM_EPOCHS, 'lr': LR,
            'lambda_repel': LAMBDA_REPEL, 'seed': SEED,
            'real_segments': int(real_n), 'fake_segments': int(fake_n),
        },
        'raw_metrics':      raw_metrics,
        'baseline_history': baseline_hist,
        'fix1a_history':    fix1a_hist,
    }

    out_path = os.path.join(SAVE_DIR, "results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Save model checkpoints
    torch.save(baseline_model.state_dict(), os.path.join(SAVE_DIR, "baseline_model.pt"))
    torch.save(fix1a_model.state_dict(),    os.path.join(SAVE_DIR, "fix1a_model.pt"))
    print("Model checkpoints saved.")


if __name__ == "__main__":
    main()
