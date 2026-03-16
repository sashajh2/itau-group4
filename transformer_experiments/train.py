"""
Training script for the vanilla transformer deepfake detector.

Usage:
    python -m transformer_experiments.train
"""

import os
import time

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

# ── Defaults ──────────────────────────────────────────────────────────────
HDF5_PATH = "exports/deepfake_embeddings.h5"
N_REAL = 300
N_FAKE = 300
TRAIN_SIZE = 400
SEED = 42

# Embeddings to use — concatenated along the feature dim.
# Available: "hubert" (768), "openl3" (512), "senet" (2048)
EMBEDDING_KEYS = ("hubert", "openl3", "senet")
EMBEDDING_DIMS  = {"hubert": 768, "openl3": 512, "senet": 2048}
INPUT_DIM = sum(EMBEDDING_DIMS[k] for k in EMBEDDING_KEYS)

EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4

MAX_SEQ_LEN = 256  # ~P95; longer sequences are truncated

D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FF = 1024
DROPOUT = 0.1
# ──────────────────────────────────────────────────────────────────────────


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Compute accuracy, precision, recall, F1 from raw logits and labels."""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in loader:
        emb = batch["embeddings"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(emb, mask).squeeze(-1)  # (B,)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(all_labels)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in loader:
        emb = batch["embeddings"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(emb, mask).squeeze(-1)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(all_labels)
    return metrics


@torch.no_grad()
def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    samples: list,
    device: torch.device,
    split_name: str,
) -> np.ndarray:
    """Detailed per-sample evaluation with confusion matrix.

    Returns:
        cm: (2, 2) confusion matrix [[TN, FP], [FN, TP]]
    """
    model.eval()
    all_logits = []
    all_labels = []

    for batch in loader:
        emb = batch["embeddings"].to(device)
        mask = batch["attention_mask"].to(device)
        logits = model(emb, mask).squeeze(-1)
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"])

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    probs = torch.sigmoid(all_logits)
    preds = (probs >= 0.5).long()
    labels_int = all_labels.long()

    tp = ((preds == 1) & (labels_int == 1)).sum().item()
    fp = ((preds == 1) & (labels_int == 0)).sum().item()
    fn = ((preds == 0) & (labels_int == 1)).sum().item()
    tn = ((preds == 0) & (labels_int == 0)).sum().item()

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    print(f"\n{'=' * 60}")
    print(f"  {split_name} Evaluation  ({len(samples)} samples)")
    print(f"{'=' * 60}")

    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Real   Fake")
    print(f"    Actual Real   {tn:4d}   {fp:4d}")
    print(f"    Actual Fake   {fn:4d}   {tp:4d}")

    print(f"\n  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1:.3f}")

    # Per-sample breakdown
    print(f"\n  Per-sample predictions:")
    print(f"  {'#':>3s}  {'True':>5s}  {'Pred':>5s}  {'Prob':>6s}  {'':>4s}  Video")
    print(f"  {'-'*55}")
    for i, s in enumerate(samples):
        true_str = "FAKE" if labels_int[i] == 1 else "REAL"
        pred_str = "FAKE" if preds[i] == 1 else "REAL"
        ok = "OK" if preds[i] == labels_int[i] else "MISS"
        print(
            f"  {i+1:3d}  {true_str:>5s}  {pred_str:>5s}  {probs[i]:.4f}  {ok:>4s}  "
            f"{s['video_id'][:30]} aug={s['aug_idx']}"
        )

    return np.array([[tn, fp], [fn, tp]])


def plot_confusion_matrices(
    train_cm: np.ndarray,
    test_cm: np.ndarray,
    save_path: str,
):
    """Plot train and test confusion matrices side-by-side and save to file."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    class_names = ["Real", "Fake"]

    for ax, cm, title in zip(axes, [train_cm, test_cm], ["Train", "Test"]):
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)

        # Annotate cells with counts and percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                color = "white" if cm[i, j] > total / 4 else "black"
                ax.text(
                    j, i, f"{cm[i, j]}\n({pct:.0f}%)",
                    ha="center", va="center", color=color, fontsize=13,
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Vanilla Transformer — Confusion Matrices", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nConfusion matrix plot saved to: {save_path}")


def fmt(metrics: dict) -> str:
    return (
        f"loss={metrics['loss']:.4f}  acc={metrics['acc']:.3f}  "
        f"prec={metrics['prec']:.3f}  rec={metrics['rec']:.3f}  f1={metrics['f1']:.3f}"
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nSampling dataset (no video overlap between train/test)...")
    n_per_class_train = TRAIN_SIZE // 2
    n_per_class_test = (N_REAL + N_FAKE - TRAIN_SIZE) // 2
    print(f"Embeddings: {' + '.join(EMBEDDING_KEYS)}  (input_dim={INPUT_DIM})")
    train_samples, test_samples = sample_train_test_no_overlap(
        HDF5_PATH,
        n_train_real=n_per_class_train,
        n_train_fake=n_per_class_train,
        n_test_real=n_per_class_test,
        n_test_fake=n_per_class_test,
        seed=SEED,
        embedding_keys=EMBEDDING_KEYS,
    )

    # Cap at MAX_SEQ_LEN — truncate outliers, pad shorter ones
    all_samples = train_samples + test_samples
    n_truncated = sum(1 for s in all_samples if s["embeddings"].shape[0] > MAX_SEQ_LEN)
    print(f"Padding/truncating all sequences to T={MAX_SEQ_LEN} ({n_truncated} truncated)")

    train_ds = DeepfakeSequenceDataset(train_samples, max_seq_len=MAX_SEQ_LEN)
    test_ds = DeepfakeSequenceDataset(test_samples, max_seq_len=MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    n_train_real = sum(1 for s in train_samples if s["label"] == 0)
    n_train_fake = sum(1 for s in train_samples if s["label"] == 1)
    n_test_real = sum(1 for s in test_samples if s["label"] == 0)
    n_test_fake = sum(1 for s in test_samples if s["label"] == 1)
    print(f"Train: {len(train_samples)} ({n_train_real} real, {n_train_fake} fake)")
    print(f"Test:  {len(test_samples)} ({n_test_real} real, {n_test_fake} fake)")

    # ── Model ─────────────────────────────────────────────────────────────
    model = VanillaTransformerClassifier(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs (batch_size={BATCH_SIZE}, lr={LR})")
    print("-" * 80)

    best_test_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        tag = ""
        if test_metrics["f1"] > best_test_f1:
            best_test_f1 = test_metrics["f1"]
            best_epoch = epoch
            tag = " *"

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  ({elapsed:.1f}s)  "
            f"train: {fmt(train_metrics)}  |  test: {fmt(test_metrics)}{tag}"
        )

    print("-" * 80)
    print(f"Best test F1: {best_test_f1:.3f} (epoch {best_epoch})")

    # ── Final Evaluation ──────────────────────────────────────────────────
    train_cm = full_evaluation(model, train_loader, train_samples, device, "TRAIN")
    test_cm = full_evaluation(model, test_loader, test_samples, device, "TEST")

    # ── Save confusion matrix plot ────────────────────────────────────────
    from datetime import datetime
    results_dir = os.path.join("results", "transformer")
    os.makedirs(results_dir, exist_ok=True)
    emb_tag = "_".join(EMBEDDING_KEYS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(results_dir, f"confusion_matrices_{emb_tag}_{timestamp}.png")
    plot_confusion_matrices(train_cm, test_cm, plot_path)


if __name__ == "__main__":
    main()
