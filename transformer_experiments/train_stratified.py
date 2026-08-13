"""Transformer experiment: stratified fake sampling across datasets.

Fake samples are split 50/50 between AVDeepFake1M and ShareVeo3, so the
model cannot overfit to a single dataset's signature.

  Train (600): 300 real (avdeepfake1m)
               150 fake  (avdeepfake1m) + 150 fake (shareveo3)
  Test  (200): 100 real (avdeepfake1m)
                50 fake  (avdeepfake1m) +  50 fake (shareveo3)

Experiments: HuBERT-only (768-dim) and OpenL3-only (512-dim).

Usage:
    python -m transformer_experiments.train_stratified
"""

import os
import random
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple

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
HDF5_PATH = "exports/deepfake_embeddings.h5"

# Real samples — all from avdeepfake1m (only source of real videos)
N_TRAIN_REAL = 300
N_TEST_REAL  = 100

# Fake samples — split 50/50 between datasets
N_TRAIN_FAKE_PER_DS = 150   # 150 avdeepfake1m + 150 shareveo3 = 300 train fake
N_TEST_FAKE_PER_DS  =  50   #  50 avdeepfake1m +  50 shareveo3 = 100 test fake

FAKE_DATASETS = ["avdeepfake1m", "shareveo3"]

SEED         = 42
EPOCHS       = 30
BATCH_SIZE   = 16
LR           = 1e-4
WEIGHT_DECAY = 1e-4
MAX_SEQ_LEN  = 256

D_MODEL    = 256
NHEAD      = 8
NUM_LAYERS = 4
DIM_FF     = 1024
DROPOUT    = 0.1

EMBEDDING_CONFIGS = [
    {"key": "senet",   "dim": 2048},
]
# ─────────────────────────────────────────────────────────────────────────────


def sample_stratified(
    hdf5_path: str,
    embedding_key: str,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Sample train/test with stratified fake representation per dataset.

    Real videos: video-level 75/25 pool split (no video overlap).
    Fake videos: split per dataset independently (75/25 pool split each),
                 then N_TRAIN/TEST_FAKE_PER_DS samples drawn from each.
    """
    rng = random.Random(seed)

    real_videos: List[Tuple[str, List[int]]] = []
    fake_by_ds: Dict[str, List[Tuple[str, List[int]]]] = {ds: [] for ds in FAKE_DATASETS}

    with h5py.File(hdf5_path, "r") as f:
        for safe_id in f["videos"].keys():
            vid = f["videos"][safe_id]
            ds = vid.attrs.get("dataset", b"")
            if isinstance(ds, bytes):
                ds = ds.decode()

            aug_types = [
                t.decode() if isinstance(t, bytes) else t
                for t in vid["augmentation_info"]["types"][:]
            ]
            real_idx = [i for i, t in enumerate(aug_types) if t in ("source", "real")]
            fake_idx = [i for i, t in enumerate(aug_types) if t == "fake"]

            if real_idx:
                real_videos.append((safe_id, real_idx))
            if fake_idx and ds in fake_by_ds:
                fake_by_ds[ds].append((safe_id, fake_idx))

    rng.shuffle(real_videos)
    for ds in FAKE_DATASETS:
        rng.shuffle(fake_by_ds[ds])

    # Split real pool 75/25
    n_real_train_pool = max(1, round(len(real_videos) * N_TRAIN_REAL / (N_TRAIN_REAL + N_TEST_REAL)))
    train_real_pool = real_videos[:n_real_train_pool]
    test_real_pool  = real_videos[n_real_train_pool:]

    # Split each fake dataset pool 75/25
    train_fake_pools: Dict[str, list] = {}
    test_fake_pools:  Dict[str, list] = {}
    for ds in FAKE_DATASETS:
        videos = fake_by_ds[ds]
        n_train = max(1, round(len(videos) * N_TRAIN_FAKE_PER_DS / (N_TRAIN_FAKE_PER_DS + N_TEST_FAKE_PER_DS)))
        train_fake_pools[ds] = videos[:n_train]
        test_fake_pools[ds]  = videos[n_train:]

    print(f"  Video pools:")
    print(f"    Real        — total: {len(real_videos)}, train: {len(train_real_pool)}, test: {len(test_real_pool)}")
    for ds in FAKE_DATASETS:
        print(f"    Fake/{ds:<14} — total: {len(fake_by_ds[ds])}, "
              f"train: {len(train_fake_pools[ds])}, test: {len(test_fake_pools[ds])}")

    def collect(pool: list, n: int, label_override: int = None) -> List[Dict]:
        samples = []
        with h5py.File(hdf5_path, "r") as f:
            for _ in range(n):
                safe_id, aug_indices = rng.choice(pool)
                aug_idx = rng.choice(aug_indices)
                vid = f["videos"][safe_id]
                emb = np.array(vid["embeddings"][embedding_key][aug_idx], dtype=np.float32)
                labels = vid["labels"]["audio"][aug_idx]
                video_label = label_override if label_override is not None else (
                    1 if np.any(labels > 0) else 0
                )
                ds_attr = vid.attrs.get("dataset", b"")
                if isinstance(ds_attr, bytes):
                    ds_attr = ds_attr.decode()
                samples.append({
                    "embeddings": emb,
                    "label": video_label,
                    "video_id": safe_id,
                    "aug_idx": int(aug_idx),
                    "dataset": ds_attr,
                })
        return samples

    train_samples = collect(train_real_pool, N_TRAIN_REAL)
    test_samples  = collect(test_real_pool,  N_TEST_REAL)

    for ds in FAKE_DATASETS:
        train_samples += collect(train_fake_pools[ds], N_TRAIN_FAKE_PER_DS)
        test_samples  += collect(test_fake_pools[ds],  N_TEST_FAKE_PER_DS)

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples


def dataset_breakdown(samples: List[Dict], split_name: str):
    print(f"\n  {split_name} breakdown (n={len(samples)}):")
    by_ds = {}
    for s in samples:
        by_ds.setdefault(s["dataset"], {"real": 0, "fake": 0})
        key = "fake" if s["label"] == 1 else "real"
        by_ds[s["dataset"]][key] += 1
    for ds, counts in sorted(by_ds.items()):
        total = counts["real"] + counts["fake"]
        print(f"    {ds:<20}  {total:4d} total  ({counts['real']} real, {counts['fake']} fake)")


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
        logits = model(batch["embeddings"].to(device),
                       batch["attention_mask"].to(device)).squeeze(-1)
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"])
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    m  = compute_metrics(logits, labels)
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    return cm, m


def plot_confusion_matrix(cm, title, ax, acc, fnr):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"], fontsize=11)
    ax.set_yticklabels(["Real", "Fake"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    total = cm.sum()
    labels_map = {(0,0):"TN",(0,1):"FP",(1,0):"FN",(1,1):"TP"}
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            color = "white" if count > total / 4 else "black"
            if (i, j) == (1, 0):
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                             linewidth=3, edgecolor="red", facecolor="none"))
            ax.text(j, i, f"{labels_map[(i,j)]}\n{count}\n({count/total*100:.0f}%)",
                    ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    ax.text(0.5, -0.18, f"Accuracy: {acc:.3f}   FNR: {fnr:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="orange", linewidth=1.5))
    return im


def save_plot(train_cm, test_cm, train_m, test_m, embedding_key, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Transformer — {embedding_key.upper()} only  |  Stratified fakes (50% avdeepfake1m / 50% shareveo3)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plot_confusion_matrix(train_cm, f"Train  (n={int(train_cm.sum())})", axes[0], train_m["acc"], train_m["fnr"])
    plot_confusion_matrix(test_cm,  f"Test  (n={int(test_cm.sum())})",  axes[1], test_m["acc"],  test_m["fnr"])
    fig.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


def run_experiment(embedding_key: str, input_dim: int, device: torch.device, results_dir: str) -> dict:
    print(f"\n{'═'*70}")
    print(f"  EXPERIMENT: {embedding_key.upper()} only  (input_dim={input_dim})")
    print(f"{'═'*70}")

    t0 = time.time()
    train_samples, test_samples = sample_stratified(HDF5_PATH, embedding_key, seed=SEED)
    print(f"  Sampling took {time.time()-t0:.1f}s")
    dataset_breakdown(train_samples, "Train")
    dataset_breakdown(test_samples,  "Test")

    train_ds = DeepfakeSequenceDataset(train_samples, max_seq_len=MAX_SEQ_LEN)
    test_ds  = DeepfakeSequenceDataset(test_samples,  max_seq_len=MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = VanillaTransformerClassifier(
        input_dim=input_dim, d_model=D_MODEL, nhead=NHEAD,
        num_layers=NUM_LAYERS, dim_feedforward=DIM_FF,
        dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"\nTraining {EPOCHS} epochs  (bs={BATCH_SIZE}, lr={LR})")
    print("-"*70)
    best_f1, best_state = 0.0, None
    t_train = time.time()

    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te = evaluate(model, test_loader, criterion, device)
        tag = ""
        if te["f1"] > best_f1:
            best_f1 = te["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tag = " *"
        print(
            f"  Ep {epoch:3d}/{EPOCHS}  ({time.time()-t_ep:.1f}s)"
            f"  train loss={tr['loss']:.4f} acc={tr['acc']:.3f} fnr={tr['fnr']:.3f}"
            f"  |  test loss={te['loss']:.4f} acc={te['acc']:.3f} fnr={te['fnr']:.3f}{tag}"
        )

    train_time = time.time() - t_train
    print("-"*70)
    print(f"  Best test F1: {best_f1:.3f}  |  Training time: {train_time:.1f}s ({train_time/60:.1f} min)")

    if best_state:
        model.load_state_dict(best_state)

    train_cm, train_m = collect_cm(model, train_loader, device)
    test_cm,  test_m  = collect_cm(model, test_loader,  device)

    for split, cm, m in [("TRAIN", train_cm, train_m), ("TEST", test_cm, test_m)]:
        print(f"\n  {split} — Accuracy: {m['acc']:.3f}   FNR: {m['fnr']:.3f}"
              f"   Precision: {m['prec']:.3f}   Recall: {m['rec']:.3f}   F1: {m['f1']:.3f}")
        print(f"           CM: [[TN={cm[0,0]}, FP={cm[0,1]}], [FN={cm[1,0]}, TP={cm[1,1]}]]")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(results_dir, f"cm_{embedding_key}_{timestamp}.png")
    save_plot(train_cm, test_cm, train_m, test_m, embedding_key, plot_path)

    return {
        "embedding":     embedding_key,
        "train_cm":      train_cm,
        "test_cm":       test_cm,
        "train_metrics": train_m,
        "test_metrics":  test_m,
        "train_time_s":  train_time,
    }


def save_summary_plot(results, results_dir):
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    fig.suptitle(
        "Transformer — Stratified fakes (50% avdeepfake1m / 50% shareveo3) — Test CMs",
        fontsize=13, fontweight="bold", y=1.02,
    )
    for ax, res in zip(axes, results):
        m = res["test_metrics"]
        plot_confusion_matrix(res["test_cm"], f"{res['embedding'].upper()} only", ax, m["acc"], m["fnr"])
    fig.tight_layout(pad=2.5)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"cm_summary_{timestamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary plot saved → {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Fake sampling: {N_TRAIN_FAKE_PER_DS} train + {N_TEST_FAKE_PER_DS} test per dataset "
          f"({', '.join(FAKE_DATASETS)})")

    results_dir = os.path.join("results", "sequence_models", "transformer_stratified")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    t_total = time.time()

    for cfg in EMBEDDING_CONFIGS:
        res = run_experiment(cfg["key"], cfg["dim"], device, results_dir)
        all_results.append(res)

    total_time = time.time() - t_total

    print(f"\n{'═'*70}")
    print(f"  RESULTS SUMMARY  (Test — {N_TEST_REAL + N_TEST_FAKE_PER_DS*2} samples)")
    print(f"{'═'*70}")
    print(f"  {'Embedding':<10}  {'Train Time':>12}  {'Accuracy':>10}  {'FNR':>8}  {'F1':>8}")
    print(f"  {'-'*55}")
    for r in all_results:
        m = r["test_metrics"]
        print(f"  {r['embedding']:<10}  {r['train_time_s']:>10.1f}s  {m['acc']:>10.3f}  {m['fnr']:>8.3f}  {m['f1']:>8.3f}")
    print(f"\n  Total wall time: {total_time:.1f}s ({total_time/60:.1f} min)")

    save_summary_plot(all_results, results_dir)


if __name__ == "__main__":
    main()
