"""
Training script for the LSTM deepfake detector on HuBERT embeddings.

Usage:
    python -m transformer_experiments.lstm_train_hub
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
from transformer_experiments.lstm_model import VideoLSTM

# ── Defaults ───────────────────────────────────────────────────────────────
HDF5_PATH = "exports/deepfake_embeddings.h5"
N_TRAIN_REAL = 200
N_TRAIN_FAKE = 200
N_TEST_REAL  = 100
N_TEST_FAKE  = 100
SEED = 42

INPUT_DIM  = 768  # HuBERT only
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT    = 0.3

EPOCHS              = 75
BATCH_SIZE          = 16
LR                  = 1e-4
EARLY_STOPPING_PAT  = 10  # stop if test F1 doesn't improve for this many epochs
# ──────────────────────────────────────────────────────────────────────────


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

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in loader:
        X      = batch["embeddings"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        lengths = mask.sum(dim=1).long()

        optimizer.zero_grad()
        logits = model(X, lengths)
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
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in loader:
        X      = batch["embeddings"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        lengths = mask.sum(dim=1).long()
        logits  = model(X, lengths)
        loss    = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(all_labels)
    return metrics


@torch.no_grad()
def full_evaluation(model, loader, samples, device, split_name):
    model.eval()
    all_logits = []
    all_labels = []

    for batch in loader:
        X      = batch["embeddings"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["labels"]

        lengths = mask.sum(dim=1).long()
        logits  = model(X, lengths)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    probs      = torch.sigmoid(all_logits)
    preds      = (probs >= 0.5).long()
    labels_int = all_labels.long()

    m = compute_metrics(all_logits, all_labels)
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]

    print(f"\n{'=' * 60}")
    print(f"  {split_name} Evaluation  ({len(samples)} samples)")
    print(f"{'=' * 60}")
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Real   Fake")
    print(f"    Actual Real   {tn:4d}   {fp:4d}")
    print(f"    Actual Fake   {fn:4d}   {tp:4d}")
    print(f"\n  Accuracy:  {m['acc']:.3f}")
    print(f"  Precision: {m['prec']:.3f}")
    print(f"  Recall:    {m['rec']:.3f}")
    print(f"  F1:        {m['f1']:.3f}")

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


def plot_training_curve(losses: list, save_path: str):
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training Loss (HuBERT)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curve saved to: {save_path}")


def plot_confusion_matrices(train_cm: np.ndarray, test_cm: np.ndarray, save_path: str):
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

        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                color = "white" if cm[i, j] > total / 4 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({pct:.0f}%)",
                        ha="center", va="center", color=color, fontsize=13)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("LSTM (HuBERT) — Confusion Matrices", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix plot saved to: {save_path}")


def fmt(metrics: dict) -> str:
    return (
        f"loss={metrics['loss']:.4f}  acc={metrics['acc']:.3f}  "
        f"prec={metrics['prec']:.3f}  rec={metrics['rec']:.3f}  f1={metrics['f1']:.3f}"
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nSampling dataset (HuBERT only, no video overlap between train/test)...")
    train_samples, test_samples = sample_train_test_no_overlap(
        HDF5_PATH,
        n_train_real=N_TRAIN_REAL,
        n_train_fake=N_TRAIN_FAKE,
        n_test_real=N_TEST_REAL,
        n_test_fake=N_TEST_FAKE,
        seed=SEED,
        embedding_keys=("hubert",),
    )

    all_lengths = [s["embeddings"].shape[0] for s in train_samples + test_samples]
    max_seq_len = max(all_lengths)
    print(f"Sequence lengths — min: {min(all_lengths)}, max: {max_seq_len}, mean: {np.mean(all_lengths):.1f}")

    train_ds = DeepfakeSequenceDataset(train_samples, max_seq_len=max_seq_len)
    test_ds  = DeepfakeSequenceDataset(test_samples,  max_seq_len=max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    n_train_real = sum(1 for s in train_samples if s["label"] == 0)
    n_train_fake = sum(1 for s in train_samples if s["label"] == 1)
    n_test_real  = sum(1 for s in test_samples  if s["label"] == 0)
    n_test_fake  = sum(1 for s in test_samples  if s["label"] == 1)
    print(f"Train: {len(train_samples)} ({n_train_real} real, {n_train_fake} fake)")
    print(f"Test:  {len(test_samples)}  ({n_test_real} real, {n_test_fake} fake)")

    # ── Model ─────────────────────────────────────────────────────────────
    model = VideoLSTM(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs (batch_size={BATCH_SIZE}, lr={LR})")
    print("-" * 80)

    losses = []
    best_test_f1  = 0.0
    best_epoch    = 0
    best_state    = None
    patience_ctr  = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics  = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        losses.append(train_metrics["loss"])

        tag = ""
        if test_metrics["f1"] > best_test_f1:
            best_test_f1 = test_metrics["f1"]
            best_epoch   = epoch
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            tag = " *"
        else:
            patience_ctr += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{EPOCHS}  ({elapsed:.1f}s)  "
                f"train: {fmt(train_metrics)}  |  test: {fmt(test_metrics)}{tag}"
            )

        if patience_ctr >= EARLY_STOPPING_PAT:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {EARLY_STOPPING_PAT} epochs)")
            break

    print("-" * 80)
    print(f"Best test F1: {best_test_f1:.3f} (epoch {best_epoch})")
    print("Restoring best model weights...")
    model.load_state_dict(best_state)

    # ── Final Evaluation ──────────────────────────────────────────────────
    train_cm = full_evaluation(model, train_loader, train_samples, device, "TRAIN")
    test_cm  = full_evaluation(model, test_loader,  test_samples,  device, "TEST")

    # ── Save plots ────────────────────────────────────────────────────────
    from datetime import datetime
    results_dir = os.path.join("results", "sequence_models", "lstm")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_training_curve(losses, os.path.join(results_dir, f"loss_{timestamp}.png"))
    plot_confusion_matrices(train_cm, test_cm, os.path.join(results_dir, f"confusion_{timestamp}.png"))


if __name__ == "__main__":
    main()
