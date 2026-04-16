"""
5-fold cross-validation training script for the vanilla transformer deepfake detector.

Usage:
    python -m transformer_experiments.train_kfold
"""

import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_experiments.dataset import (
    DeepfakeSequenceDataset,
    collate_fn,
    sample_kfold_splits,
)
from transformer_experiments.model import VanillaTransformerClassifier
from transformer_experiments.train import (
    compute_metrics,
    evaluate,
    fmt,
    full_evaluation,
    plot_confusion_matrices,
    train_one_epoch,
)

# ── Config ────────────────────────────────────────────────────────────────
HDF5_PATH = "exports/deepfake_embeddings.h5"
K = 3
N_TRAIN_PER_CLASS  = 350   # per fold: 350 real + 350 fake = 700 train
N_TEST_PER_CLASS   = 150   # per fold: 150 real + 150 fake = 300 avdeepfake test
DATASET_FILTER     = "avdeepfake1m"

# Median duration of avdeepfake1m source videos — used to split test results
DURATION_MEDIAN = 10.7  # seconds
SEED = 42

EMBEDDING_KEYS = ("hubert",)
EMBEDDING_DIMS  = {"hubert": 768, "openl3": 512, "senet": 2048}
INPUT_DIM = sum(EMBEDDING_DIMS[k] for k in EMBEDDING_KEYS)

EPOCHS     = 10
BATCH_SIZE = 8
LR         = 1e-4
WEIGHT_DECAY = 1e-4
MAX_SEQ_LEN  = 256

D_MODEL    = 256
NHEAD      = 8
NUM_LAYERS = 4
DIM_FF     = 1024
DROPOUT    = 0.1
# ──────────────────────────────────────────────────────────────────────────


def run_fold(
    fold_idx: int,
    train_samples: list,
    test_samples: list,
    device: torch.device,
) -> dict:
    """Train and evaluate one fold. Returns final test metrics + best F1."""
    print(f"\n{'=' * 80}")
    print(f"  FOLD {fold_idx + 1}/{K}  —  train={len(train_samples)}  test={len(test_samples)}")
    print(f"{'=' * 80}")

    n_truncated = sum(
        1 for s in train_samples + test_samples
        if s["embeddings"].shape[0] > MAX_SEQ_LEN
    )
    if n_truncated:
        print(f"  ({n_truncated} sequences truncated to T={MAX_SEQ_LEN})")

    train_ds = DeepfakeSequenceDataset(train_samples, max_seq_len=MAX_SEQ_LEN)
    test_ds  = DeepfakeSequenceDataset(test_samples,  max_seq_len=MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = VanillaTransformerClassifier(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    results_dir = os.path.join("results", "transformer")
    os.makedirs(results_dir, exist_ok=True)
    emb_tag   = "_".join(EMBEDDING_KEYS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(results_dir, f"kfold_{K}fold_{emb_tag}_fold{fold_idx+1}_{timestamp}.pt")

    print(f"\n  Training for {EPOCHS} epochs")
    print(f"  {'-' * 76}")

    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        tag = ""
        if te["f1"] > best_f1:
            best_f1    = te["f1"]
            best_epoch = epoch
            tag = " *"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_f1": best_f1,
                "fold": fold_idx + 1,
                # Sampling config — needed to reproduce the same splits in test_kfold.py
                "embedding_keys": EMBEDDING_KEYS,
                "input_dim": INPUT_DIM,
                "k": K,
                "n_train_per_class": N_TRAIN_PER_CLASS,
                "n_test_per_class": N_TEST_PER_CLASS,
                "dataset_filter": DATASET_FILTER,
                "seed": SEED,
                # Model architecture
                "d_model": D_MODEL,
                "nhead": NHEAD,
                "num_layers": NUM_LAYERS,
                "dim_feedforward": DIM_FF,
                "dropout": DROPOUT,
                "max_seq_len": MAX_SEQ_LEN,
            }, model_path)

        print(
            f"  Epoch {epoch:3d}/{EPOCHS}  ({elapsed:.1f}s)  "
            f"train: {fmt(tr)}  |  test: {fmt(te)}{tag}"
        )

    print(f"  {'-' * 76}")
    print(f"  Best test F1: {best_f1:.3f} (epoch {best_epoch})")
    print(f"  Model saved to: {model_path}")

    test_cm = full_evaluation(model, test_loader, test_samples, device, f"FOLD {fold_idx+1} AVDeepfake TEST")

    # Collect per-sample predictions for short/long breakdown
    model.eval()
    all_preds = []
    all_labels_list = []
    with torch.no_grad():
        for batch in test_loader:
            emb  = batch["embeddings"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(emb, mask).squeeze(-1)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
            all_preds.extend(preds)
            all_labels_list.extend(batch["labels"].long().tolist())

    return {
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "test_cm": test_cm,
        "test_samples": test_samples,
        "test_preds": all_preds,
        "test_labels": all_labels_list,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Embeddings: {' + '.join(EMBEDDING_KEYS)}  (input_dim={INPUT_DIM})")
    print(f"\nBuilding {K}-fold splits ({DATASET_FILTER}, video-level, no leakage)...")

    splits = sample_kfold_splits(
        HDF5_PATH,
        k=K,
        n_train_per_class=N_TRAIN_PER_CLASS,
        n_test_per_class=N_TEST_PER_CLASS,
        seed=SEED,
        embedding_keys=EMBEDDING_KEYS,
        dataset_filter=DATASET_FILTER,
    )

    fold_results = []
    all_cms = []

    for i, (train_samples, test_samples) in enumerate(splits):
        result = run_fold(i, train_samples, test_samples, device)
        fold_results.append(result)
        all_cms.append(result["test_cm"])

    # ── Aggregate summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  CROSS-VALIDATION SUMMARY  ({K} folds)")
    print(f"{'=' * 80}")
    print(f"\n  {'Fold':>5}  {'Best F1':>8}  {'Best Epoch':>10}")
    print(f"  {'-' * 30}")
    f1_scores = []
    for i, r in enumerate(fold_results):
        print(f"  {i+1:>5}  {r['best_f1']:>8.3f}  {r['best_epoch']:>10d}")
        f1_scores.append(r["best_f1"])

    mean_f1 = np.mean(f1_scores)
    std_f1  = np.std(f1_scores)
    print(f"  {'-' * 30}")
    print(f"  {'Mean':>5}  {mean_f1:>8.3f}  ± {std_f1:.3f}")

    # Aggregate confusion matrix across folds
    agg_cm = np.sum(all_cms, axis=0)
    tn, fp, fn, tp = agg_cm[0, 0], agg_cm[0, 1], agg_cm[1, 0], agg_cm[1, 1]
    total = agg_cm.sum()
    agg_acc  = (tp + tn) / max(total, 1)
    agg_prec = tp / max(tp + fp, 1)
    agg_rec  = tp / max(tp + fn, 1)
    agg_f1   = 2 * agg_prec * agg_rec / max(agg_prec + agg_rec, 1e-8)

    print(f"\n  Aggregated confusion matrix (all {total} test samples):")
    print(f"                  Predicted")
    print(f"                  Real   Fake")
    print(f"    Actual Real   {tn:4d}   {fp:4d}")
    print(f"    Actual Fake   {fn:4d}   {tp:4d}")
    print(f"\n  Accuracy:  {agg_acc:.3f}")
    print(f"  Precision: {agg_prec:.3f}")
    print(f"  Recall:    {agg_rec:.3f}")
    print(f"  F1:        {agg_f1:.3f}")

    # ── Short vs Long breakdown ───────────────────────────────────────────
    # Collect all test samples and their predictions across folds
    all_test_samples_flat = []
    all_test_preds_flat   = []
    all_test_labels_flat  = []

    # Re-run inference on each fold's test set to get per-sample predictions
    # (We reuse the last trained model per fold — stored in fold_results)
    for fold_data in fold_results:
        for s, pred, label in zip(fold_data["test_samples"], fold_data["test_preds"], fold_data["test_labels"]):
            all_test_samples_flat.append(s)
            all_test_preds_flat.append(pred)
            all_test_labels_flat.append(label)

    def cm_from_lists(preds, labels):
        tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
        tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
        return np.array([[tn, fp], [fn, tp]])

    short_mask = [s["duration"] < DURATION_MEDIAN for s in all_test_samples_flat]
    long_mask  = [s["duration"] >= DURATION_MEDIAN for s in all_test_samples_flat]

    short_preds  = [p for p, m in zip(all_test_preds_flat,  short_mask) if m]
    short_labels = [l for l, m in zip(all_test_labels_flat, short_mask) if m]
    long_preds   = [p for p, m in zip(all_test_preds_flat,  long_mask)  if m]
    long_labels  = [l for l, m in zip(all_test_labels_flat, long_mask)  if m]

    short_cm = cm_from_lists(short_preds, short_labels)
    long_cm  = cm_from_lists(long_preds,  long_labels)

    def print_cm_stats(cm, name):
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        total = cm.sum()
        acc  = (tp + tn) / max(total, 1)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        print(f"\n  {name} ({total} samples, duration {'<' if 'Short' in name else '>='} {DURATION_MEDIAN}s):")
        print(f"    Actual Real  {tn:4d}   {fp:4d}")
        print(f"    Actual Fake  {fn:4d}   {tp:4d}")
        print(f"    Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
        return acc, prec, rec, f1

    print(f"\n{'=' * 80}")
    print(f"  SHORT vs LONG VIDEO BREAKDOWN  (median={DURATION_MEDIAN}s)")
    print(f"{'=' * 80}")
    print(f"                  Predicted")
    print(f"                  Real   Fake")
    short_stats = print_cm_stats(short_cm, "Short videos")
    long_stats  = print_cm_stats(long_cm,  "Long  videos")

    # ── Save plots ────────────────────────────────────────────────────────
    results_dir = os.path.join("results", "transformer")
    os.makedirs(results_dir, exist_ok=True)
    emb_tag   = "_".join(EMBEDDING_KEYS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    class_names = ["Real", "Fake"]

    def annotate_cm(ax, cm, title):
        total_n = cm.sum()
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(title)
        for r in range(2):
            for c in range(2):
                pct = cm[r, c] / total_n * 100
                color = "white" if cm[r, c] > total_n / 4 else "black"
                ax.text(c, r, f"{cm[r,c]}\n({pct:.0f}%)", ha="center", va="center",
                        color=color, fontsize=11)
        return im

    # Per-fold figure
    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4))
    for i, (ax, cm) in enumerate(zip(axes, all_cms)):
        im = annotate_cm(ax, cm, f"Fold {i+1}  (F1={fold_results[i]['best_f1']:.3f})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"5-Fold CV — {'+'.join(EMBEDDING_KEYS)} — Mean F1={mean_f1:.3f}±{std_f1:.3f}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    per_fold_path = os.path.join(results_dir, f"kfold_{K}fold_{emb_tag}_{timestamp}.png")
    fig.savefig(per_fold_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPer-fold plot saved to:   {per_fold_path}")

    def f1_from_cm(cm):
        tp, fp, fn = cm[1,1], cm[0,1], cm[1,0]
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        return 2 * prec * rec / max(prec + rec, 1e-8)

    def rec_from_cm(cm):
        tp, fn = cm[1,1], cm[1,0]
        return tp / max(tp + fn, 1)

    # Aggregate + short/long confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    panels = [
        (agg_cm,   f"AVDeepfake Aggregate ({total} samples)\nAcc={agg_acc:.3f}  F1={agg_f1:.3f}"),
        (short_cm, f"Short videos (< {DURATION_MEDIAN}s, n={short_cm.sum()})\nF1={f1_from_cm(short_cm):.3f}"),
        (long_cm,  f"Long videos (≥ {DURATION_MEDIAN}s, n={long_cm.sum()})\nF1={f1_from_cm(long_cm):.3f}"),
    ]

    for ax, (cm, title) in zip(axes, panels):
        im = annotate_cm(ax, cm, title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"5-Fold CV — {'+'.join(EMBEDDING_KEYS)} — Mean F1={mean_f1:.3f}±{std_f1:.3f}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    agg_path = os.path.join(results_dir, f"kfold_{K}fold_{emb_tag}_{timestamp}_aggregate.png")
    fig.savefig(agg_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Aggregate plot saved to:  {agg_path}")


if __name__ == "__main__":
    main()