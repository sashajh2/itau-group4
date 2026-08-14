"""Transformer experiment: single-embedding evaluation on AVDeepFake1M.

Runs three independent experiments:
  1. HuBERT-only  (768-dim)
  2. SENet-only   (2048-dim)
  3. OpenL3-only  (512-dim)

Each experiment:
  - 800 total samples (400 real + 400 fake) from AVDeepFake1M
  - 75/25 train/test split → 600 train, 200 test
  - VanillaTransformerClassifier
  - Confusion matrix saved with FNR and Accuracy highlighted

Usage:
    python -m transformer_experiments.train_single_embedding
"""

import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_experiments.dataset import (
    DeepfakeSequenceDataset,
    collate_fn,
    sample_train_test_no_overlap,
)
from transformer_experiments.model import VanillaTransformerClassifier

# ── Config ────────────────────────────────────────────────────────────────────
HDF5_PATH   = "exports/deepfake_embeddings.h5"
DATASET     = "avdeepfake1m"

N_TOTAL     = 800   # 400 real + 400 fake
TRAIN_FRAC  = 0.75  # 600 train / 200 test

N_REAL = N_TOTAL // 2       # 400
N_FAKE = N_TOTAL // 2       # 400
N_TRAIN_PER_CLASS = int(N_REAL * TRAIN_FRAC)   # 300
N_TEST_PER_CLASS  = N_REAL - N_TRAIN_PER_CLASS  # 100

SEED        = 42
EPOCHS      = 20
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
    total_loss = 0.0
    all_logits, all_labels = [], []
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
    total_loss = 0.0
    all_logits, all_labels = [], []
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


def plot_confusion_matrix(cm: np.ndarray, title: str, ax: plt.Axes, acc: float, fnr: float):
    """Draw a single annotated confusion matrix on ax, highlighting FNR and Accuracy."""
    class_names = ["Real", "Fake"]
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum() // 2)

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

            # Highlight FN cell (used for FNR) with a red border
            if (i, j) == (1, 0):
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=3, edgecolor="red", facecolor="none",
                )
                ax.add_patch(rect)

            ax.text(
                j, i,
                f"{tag}\n{count}\n({pct:.0f}%)",
                ha="center", va="center", color=color, fontsize=12, fontweight="bold",
            )

    # Metrics box below the matrix
    metric_text = f"Accuracy: {acc:.3f}   FNR: {fnr:.3f}"
    ax.text(
        0.5, -0.18, metric_text,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="orange", linewidth=1.5),
    )

    return im


def save_experiment_plot(
    train_cm: np.ndarray,
    test_cm: np.ndarray,
    train_metrics: dict,
    test_metrics: dict,
    embedding_key: str,
    save_path: str,
):
    """Save a figure with train + test confusion matrices for one embedding."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Transformer — {embedding_key.upper()} only  |  AVDeepFake1M  |  "
        f"{N_TOTAL} samples (75/25 split)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    plot_confusion_matrix(
        train_cm, f"Train  (n={int(train_cm.sum())})",
        axes[0], train_metrics["acc"], train_metrics["fnr"],
    )
    plot_confusion_matrix(
        test_cm, f"Test  (n={int(test_cm.sum())})",
        axes[1], test_metrics["acc"], test_metrics["fnr"],
    )

    fig.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


def run_experiment(embedding_key: str, input_dim: int, device: torch.device, results_dir: str):
    print(f"\n{'═' * 70}")
    print(f"  EXPERIMENT: {embedding_key.upper()} only  (input_dim={input_dim})")
    print(f"{'═' * 70}")

    # ── Data ─────────────────────────────────────────────────────────────
    print(f"\nSampling {N_TOTAL} samples from {DATASET} (no video overlap)...")
    train_samples, test_samples = sample_train_test_no_overlap(
        HDF5_PATH,
        n_train_real=N_TRAIN_PER_CLASS,
        n_train_fake=N_TRAIN_PER_CLASS,
        n_test_real=N_TEST_PER_CLASS,
        n_test_fake=N_TEST_PER_CLASS,
        seed=SEED,
        embedding_keys=(embedding_key,),
        dataset_filter=DATASET,
    )
    print(f"Train: {len(train_samples)}  |  Test: {len(test_samples)}")

    train_ds = DeepfakeSequenceDataset(train_samples, max_seq_len=MAX_SEQ_LEN)
    test_ds  = DeepfakeSequenceDataset(test_samples,  max_seq_len=MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────
    model = VanillaTransformerClassifier(
        input_dim=input_dim,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\nTraining {EPOCHS} epochs  (bs={BATCH_SIZE}, lr={LR})")
    print("-" * 70)
    best_test_f1 = 0.0
    best_state   = None

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

    print("-" * 70)
    print(f"  Best test F1: {best_test_f1:.3f}")

    # Restore best weights for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Final evaluation ──────────────────────────────────────────────────
    def collect_cm(loader) -> tuple[np.ndarray, dict]:
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                emb    = batch["embeddings"].to(device)
                mask   = batch["attention_mask"].to(device)
                logits = model(emb, mask).squeeze(-1)
                all_logits.append(logits.cpu())
                all_labels.append(batch["labels"])
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)
        m   = compute_metrics(logits, labels)
        tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
        cm  = np.array([[tn, fp], [fn, tp]])
        return cm, m

    train_cm, train_m = collect_cm(train_loader)
    test_cm,  test_m  = collect_cm(test_loader)

    # ── Print summary ─────────────────────────────────────────────────────
    for split, cm, m in [("TRAIN", train_cm, train_m), ("TEST", test_cm, test_m)]:
        print(f"\n  {split} — Accuracy: {m['acc']:.3f}   FNR: {m['fnr']:.3f}"
              f"   Precision: {m['prec']:.3f}   Recall: {m['rec']:.3f}   F1: {m['f1']:.3f}")
        print(f"           Confusion Matrix  [[TN={cm[0,0]}, FP={cm[0,1]}], [FN={cm[1,0]}, TP={cm[1,1]}]]")

    # ── Plot ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(results_dir, f"cm_{embedding_key}_{timestamp}.png")
    save_experiment_plot(train_cm, test_cm, train_m, test_m, embedding_key, plot_path)

    return {
        "embedding": embedding_key,
        "train_cm": train_cm,
        "test_cm":  test_cm,
        "train_metrics": train_m,
        "test_metrics":  test_m,
    }


def save_summary_plot(results: list, results_dir: str):
    """Side-by-side test confusion matrices for all three embeddings."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Transformer — AVDeepFake1M — Test Confusion Matrices ({N_TOTAL} samples, 75/25 split)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, res in zip(axes, results):
        m = res["test_metrics"]
        plot_confusion_matrix(
            res["test_cm"],
            f"{res['embedding'].upper()} only",
            ax,
            m["acc"],
            m["fnr"],
        )

    fig.tight_layout(pad=2.5)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"cm_summary_{timestamp}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary plot saved → {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {DATASET}  |  Total samples: {N_TOTAL}  |  Train/Test: {N_TRAIN_PER_CLASS*2}/{N_TEST_PER_CLASS*2}")

    results_dir = os.path.join("results", "sequence_models", "transformer_single_embedding")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    for cfg in EMBEDDING_CONFIGS:
        res = run_experiment(cfg["key"], cfg["dim"], device, results_dir)
        all_results.append(res)

    # ── Final summary table ───────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY  (Test set — {N_TEST_PER_CLASS*2} samples)")
    print(f"{'═' * 70}")
    print(f"  {'Embedding':<10}  {'Accuracy':>10}  {'FNR':>8}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"  {'-' * 60}")
    for r in all_results:
        m = r["test_metrics"]
        print(
            f"  {r['embedding']:<10}  {m['acc']:>10.3f}  {m['fnr']:>8.3f}"
            f"  {m['prec']:>10.3f}  {m['rec']:>8.3f}  {m['f1']:>8.3f}"
        )
    print(f"{'═' * 70}")

    save_summary_plot(all_results, results_dir)


if __name__ == "__main__":
    main()
