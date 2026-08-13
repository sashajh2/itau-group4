"""Transformer experiment: single-embedding evaluation across ALL datasets.

No dataset filter is applied — the video pool includes every dataset present
in the HDF5 file (avdeepfake1m, shareveo3, sora2, etc.).

Runs three independent experiments:
  1. HuBERT-only  (768-dim)
  2. SENet-only   (2048-dim)
  3. OpenL3-only  (512-dim)

Each experiment:
  - 800 total samples (400 real + 400 fake) drawn from all datasets
  - 75/25 train/test split → 600 train, 200 test (video-level no-overlap)
  - VanillaTransformerClassifier
  - Reports: time taken, accuracy, FNR, per-dataset sample breakdown

Usage:
    python -m transformer_experiments.train_all_datasets
"""

import os
import random
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_experiments.dataset import DeepfakeSequenceDataset, collate_fn
from transformer_experiments.model import VanillaTransformerClassifier

# ── Config ────────────────────────────────────────────────────────────────────
HDF5_PATH  = "exports/deepfake_embeddings.h5"

N_REAL = 400   # 400 real + 400 fake = 800 total
N_FAKE = 400
N_TRAIN_PER_CLASS = 300   # 75%
N_TEST_PER_CLASS  = 100   # 25%

SEED        = 42
EPOCHS      = 30
BATCH_SIZE  = 16
LR          = 1e-4
WEIGHT_DECAY = 1e-4
MAX_SEQ_LEN = 256

D_MODEL    = 256
NHEAD      = 8
NUM_LAYERS = 4
DIM_FF     = 1024
DROPOUT    = 0.1

EMBEDDING_CONFIGS = [
    {"key": "hubert",  "dim": 768},
    {"key": "senet",   "dim": 2048},
    {"key": "openl3",  "dim": 512},
]
# ─────────────────────────────────────────────────────────────────────────────


def sample_with_dataset_tracking(
    hdf5_path: str,
    n_train_real: int,
    n_train_fake: int,
    n_test_real: int,
    n_test_fake: int,
    embedding_key: str,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Sample train/test sets with strict video-level separation across ALL datasets.

    Identical logic to sample_train_test_no_overlap but additionally records
    the 'dataset' attribute on every sample for breakdown reporting.
    No dataset filter is applied.
    """
    rng = random.Random(seed)

    real_videos: List[Tuple[str, List[int]]] = []
    fake_videos: List[Tuple[str, List[int]]] = []

    with h5py.File(hdf5_path, "r") as f:
        for safe_id in f["videos"].keys():
            vid = f["videos"][safe_id]
            aug_types = [
                t.decode() if isinstance(t, bytes) else t
                for t in vid["augmentation_info"]["types"][:]
            ]
            real_idx = [i for i, t in enumerate(aug_types) if t in ("source", "real")]
            fake_idx = [i for i, t in enumerate(aug_types) if t == "fake"]
            if real_idx:
                real_videos.append((safe_id, real_idx))
            if fake_idx:
                fake_videos.append((safe_id, fake_idx))

    rng.shuffle(real_videos)
    rng.shuffle(fake_videos)

    real_train_ratio = n_train_real / (n_train_real + n_test_real)
    n_real_train_pool = max(1, round(len(real_videos) * real_train_ratio))
    train_real_pool = real_videos[:n_real_train_pool]
    test_real_pool  = real_videos[n_real_train_pool:]

    fake_train_ratio = n_train_fake / (n_train_fake + n_test_fake)
    n_fake_train_pool = max(1, round(len(fake_videos) * fake_train_ratio))
    train_fake_pool = fake_videos[:n_fake_train_pool]
    test_fake_pool  = fake_videos[n_fake_train_pool:]

    print(f"  All-dataset video index:")
    print(f"    Real videos — total: {len(real_videos)}, train pool: {len(train_real_pool)}, test pool: {len(test_real_pool)}")
    print(f"    Fake videos — total: {len(fake_videos)}, train pool: {len(train_fake_pool)}, test pool: {len(test_fake_pool)}")

    def collect(pool: list, n: int) -> List[Dict]:
        samples = []
        with h5py.File(hdf5_path, "r") as f:
            for _ in range(n):
                safe_id, aug_indices = rng.choice(pool)
                aug_idx = rng.choice(aug_indices)
                vid = f["videos"][safe_id]
                emb = np.array(vid["embeddings"][embedding_key][aug_idx], dtype=np.float32)
                labels = vid["labels"]["audio"][aug_idx]
                video_label = 1 if np.any(labels > 0) else 0
                ds = vid.attrs.get("dataset", b"")
                if isinstance(ds, bytes):
                    ds = ds.decode()
                samples.append({
                    "embeddings": emb,
                    "label": video_label,
                    "video_id": safe_id,
                    "aug_idx": int(aug_idx),
                    "dataset": ds,
                })
        return samples

    train_samples = collect(train_real_pool, n_train_real) + collect(train_fake_pool, n_train_fake)
    test_samples  = collect(test_real_pool,  n_test_real)  + collect(test_fake_pool,  n_test_fake)
    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples


def dataset_breakdown(samples: List[Dict], split_name: str):
    """Print a per-dataset breakdown of sample counts and class distribution."""
    ds_counter = Counter(s["dataset"] for s in samples)
    print(f"\n  {split_name} dataset breakdown (n={len(samples)}):")
    for ds, count in sorted(ds_counter.items()):
        real_n = sum(1 for s in samples if s["dataset"] == ds and s["label"] == 0)
        fake_n = sum(1 for s in samples if s["dataset"] == ds and s["label"] == 1)
        print(f"    {ds:<20}  {count:4d} samples  ({real_n} real, {fake_n} fake)")


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    fnr  = fn / max(fn + tp, 1)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "fnr": fnr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, all_logits, all_labels = 0.0, [], []
    for batch in loader:
        emb    = batch["embeddings"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(emb, mask).squeeze(-1)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    m = compute_metrics(all_logits, all_labels)
    m["loss"] = total_loss / len(all_labels)
    return m


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []
    for batch in loader:
        emb    = batch["embeddings"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(emb, mask).squeeze(-1)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    m = compute_metrics(all_logits, all_labels)
    m["loss"] = total_loss / len(all_labels)
    return m


@torch.no_grad()
def collect_cm(model, loader, device) -> Tuple[np.ndarray, dict]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        emb    = batch["embeddings"].to(device)
        mask   = batch["attention_mask"].to(device)
        logits = model(emb, mask).squeeze(-1)
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"])
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    m  = compute_metrics(logits, labels)
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    return cm, m


def plot_confusion_matrix(cm: np.ndarray, title: str, ax: plt.Axes, acc: float, fnr: float):
    class_names = ["Real", "Fake"]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    total = cm.sum()
    labels_map = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = count / total * 100
            tag   = labels_map[(i, j)]
            color = "white" if count > total / 4 else "black"
            if (i, j) == (1, 0):
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     linewidth=3, edgecolor="red", facecolor="none")
                ax.add_patch(rect)
            ax.text(j, i, f"{tag}\n{count}\n({pct:.0f}%)",
                    ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    ax.text(0.5, -0.18, f"Accuracy: {acc:.3f}   FNR: {fnr:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="orange", linewidth=1.5))
    return im


def save_experiment_plot(train_cm, test_cm, train_m, test_m, embedding_key, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Transformer — {embedding_key.upper()} only  |  All Datasets  |  "
        f"{N_REAL + N_FAKE} samples (75/25 split)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plot_confusion_matrix(train_cm, f"Train  (n={int(train_cm.sum())})",
                          axes[0], train_m["acc"], train_m["fnr"])
    plot_confusion_matrix(test_cm,  f"Test  (n={int(test_cm.sum())})",
                          axes[1], test_m["acc"], test_m["fnr"])
    fig.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


def run_experiment(embedding_key: str, input_dim: int, device: torch.device,
                   results_dir: str) -> dict:
    print(f"\n{'═' * 70}")
    print(f"  EXPERIMENT: {embedding_key.upper()} only  (input_dim={input_dim})")
    print(f"{'═' * 70}")

    # ── Data ─────────────────────────────────────────────────────────────
    print(f"\nSampling {N_REAL + N_FAKE} samples from ALL datasets...")
    t_sample = time.time()
    train_samples, test_samples = sample_with_dataset_tracking(
        HDF5_PATH,
        n_train_real=N_TRAIN_PER_CLASS,
        n_train_fake=N_TRAIN_PER_CLASS,
        n_test_real=N_TEST_PER_CLASS,
        n_test_fake=N_TEST_PER_CLASS,
        embedding_key=embedding_key,
        seed=SEED,
    )
    print(f"  Sampling took {time.time() - t_sample:.1f}s")

    dataset_breakdown(train_samples, "Train")
    dataset_breakdown(test_samples,  "Test")

    train_ds = DeepfakeSequenceDataset(train_samples, max_seq_len=MAX_SEQ_LEN)
    test_ds  = DeepfakeSequenceDataset(test_samples,  max_seq_len=MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────
    model = VanillaTransformerClassifier(
        input_dim=input_dim, d_model=D_MODEL, nhead=NHEAD,
        num_layers=NUM_LAYERS, dim_feedforward=DIM_FF,
        dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\nTraining {EPOCHS} epochs  (bs={BATCH_SIZE}, lr={LR})")
    print("-" * 70)
    best_test_f1 = 0.0
    best_state   = None
    t_train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        tag = ""
        if te["f1"] > best_test_f1:
            best_test_f1 = te["f1"]
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tag = " *"

        print(
            f"  Ep {epoch:3d}/{EPOCHS}  ({elapsed:.1f}s)"
            f"  train loss={tr['loss']:.4f} acc={tr['acc']:.3f} fnr={tr['fnr']:.3f}"
            f"  |  test loss={te['loss']:.4f} acc={te['acc']:.3f} fnr={te['fnr']:.3f}{tag}"
        )

    total_train_time = time.time() - t_train_start
    print("-" * 70)
    print(f"  Best test F1: {best_test_f1:.3f}")
    print(f"  Total training time: {total_train_time:.1f}s ({total_train_time/60:.1f} min)")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Final evaluation ──────────────────────────────────────────────────
    train_cm, train_m = collect_cm(model, train_loader, device)
    test_cm,  test_m  = collect_cm(model, test_loader,  device)

    for split, cm, m in [("TRAIN", train_cm, train_m), ("TEST", test_cm, test_m)]:
        print(f"\n  {split} — Accuracy: {m['acc']:.3f}   FNR: {m['fnr']:.3f}"
              f"   Precision: {m['prec']:.3f}   Recall: {m['rec']:.3f}   F1: {m['f1']:.3f}")
        print(f"           CM: [[TN={cm[0,0]}, FP={cm[0,1]}], [FN={cm[1,0]}, TP={cm[1,1]}]]")

    # ── Plot ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(results_dir, f"cm_{embedding_key}_{timestamp}.png")
    save_experiment_plot(train_cm, test_cm, train_m, test_m, embedding_key, plot_path)

    return {
        "embedding":       embedding_key,
        "train_cm":        train_cm,
        "test_cm":         test_cm,
        "train_metrics":   train_m,
        "test_metrics":    test_m,
        "train_time_s":    total_train_time,
        "train_breakdown": Counter(s["dataset"] for s in train_samples),
        "test_breakdown":  Counter(s["dataset"] for s in test_samples),
    }


def save_summary_plot(results: list, results_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Transformer — All Datasets — Test Confusion Matrices "
        f"({N_REAL + N_FAKE} samples, 75/25 split)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    for ax, res in zip(axes, results):
        m = res["test_metrics"]
        plot_confusion_matrix(res["test_cm"], f"{res['embedding'].upper()} only",
                              ax, m["acc"], m["fnr"])
    fig.tight_layout(pad=2.5)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"cm_summary_{timestamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary plot saved → {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Total samples: {N_REAL + N_FAKE}  |  Train: {N_TRAIN_PER_CLASS*2}  |  Test: {N_TEST_PER_CLASS*2}")
    print(f"Dataset filter: NONE (all datasets)")

    results_dir = os.path.join("results", "sequence_models", "transformer_all_datasets")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    experiment_start = time.time()

    for cfg in EMBEDDING_CONFIGS:
        res = run_experiment(cfg["key"], cfg["dim"], device, results_dir)
        all_results.append(res)

    total_time = time.time() - experiment_start

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  RESULTS SUMMARY  (Test set — {N_TEST_PER_CLASS*2} samples)")
    print(f"{'═' * 70}")
    print(f"  {'Embedding':<10}  {'Train Time':>12}  {'Accuracy':>10}  {'FNR':>8}  {'F1':>8}")
    print(f"  {'-' * 55}")
    for r in all_results:
        m = r["test_metrics"]
        t = r["train_time_s"]
        print(f"  {r['embedding']:<10}  {t:>10.1f}s  {m['acc']:>10.3f}  {m['fnr']:>8.3f}  {m['f1']:>8.3f}")

    print(f"\n  Total wall time: {total_time:.1f}s ({total_time/60:.1f} min)")

    print(f"\n{'═' * 70}")
    print(f"  DATASET BREAKDOWN")
    print(f"{'═' * 70}")
    for r in all_results:
        print(f"\n  [{r['embedding'].upper()}]")
        all_ds = sorted(set(list(r["train_breakdown"].keys()) + list(r["test_breakdown"].keys())))
        print(f"  {'Dataset':<22}  {'Train':>8}  {'Test':>8}")
        print(f"  {'-' * 42}")
        for ds in all_ds:
            print(f"  {ds:<22}  {r['train_breakdown'].get(ds, 0):>8}  {r['test_breakdown'].get(ds, 0):>8}")

    save_summary_plot(all_results, results_dir)


if __name__ == "__main__":
    main()
