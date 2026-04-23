"""
Training script for the LSTM deepfake detector on a pair of concatenated embeddings.

To switch embedding pair, uncomment the desired block under "Embedding pair config"
and comment out the others. Three pairs are available:
    HuBERT + OpenL3  -> 768 + 512  = 1280-dim
    HuBERT + SENet   -> 768 + 2048 = 2816-dim
    OpenL3 + SENet   -> 512 + 2048 = 2560-dim

Usage:
    python -m transformer_experiments.lstm_train_pair
"""

import os
import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer_experiments.dataset import (
    DeepfakeSequenceDataset,
    collate_fn,
    sample_cross_dataset_split,
    sample_train_test_no_overlap,
)
# To switch LSTM directionality, swap the import below:
#   Bidirectional  -> from transformer_experiments.lstm_model import VideoLSTM
#   Unidirectional -> from transformer_experiments.lstm_baseline_model import VideoLSTMBaseline as VideoLSTM
from transformer_experiments.lstm_model import VideoLSTM
# ── Defaults ───────────────────────────────────────────────────────────────
HDF5_PATH = "exports/deepfake_embeddings.h5"
N_TRAIN_REAL = 200
N_TRAIN_FAKE = 200
N_TEST_REAL  = 100
N_TEST_FAKE  = 100
SEED = 42

# ── Embedding pair config ──────────────────────────────────────────────────
# Uncomment exactly ONE block below.

# HuBERT + OpenL3 (1280-dim)
# EMBEDDING_KEYS = ("hubert", "openl3")
# INPUT_DIM = 768 + 512  # 1280

# HuBERT + SENet (2816-dim)
# EMBEDDING_KEYS = ("hubert", "senet")
# INPUT_DIM = 768 + 2048  # 2816

# OpenL3 + SENet (2560-dim)
EMBEDDING_KEYS = ("openl3", "senet")
INPUT_DIM = 512 + 2048  # 2560
# ──────────────────────────────────────────────────────────────────────────

HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT    = 0.3

EPOCHS              = 75
BATCH_SIZE          = 16
LR                  = 1e-4
EARLY_STOPPING_PAT  = 10
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
    f2   = 5 * prec * rec / max(4 * prec + rec, 1e-8)

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "f2": f2,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


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

    t_start = time.time()
    for batch in loader:
        X      = batch["embeddings"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["labels"]

        lengths = mask.sum(dim=1).long()
        logits  = model(X, lengths)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    t_end = time.time()

    speed = len(samples) / max(t_end - t_start, 1e-6)  # samples/sec

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
    print(f"  F2:        {m['f2']:.3f}")
    print(f"  Speed:     {speed:.1f} samples/sec")

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

    cm = np.array([[tn, fp], [fn, tp]])
    return cm, m, speed


def plot_training_curve(losses: list, title: str, save_path: str):
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"LSTM Training Loss — {title}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curve saved to: {save_path}")


def plot_confusion_matrices(
    train_cm: np.ndarray,
    test_cm: np.ndarray,
    train_metrics: dict,
    test_metrics: dict,
    train_speed: float,
    test_speed: float,
    title: str,
    save_path: str,
):
    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.45, wspace=0.35)

    class_names = ["Real", "Fake"]
    splits = [("Train", train_cm, train_metrics, train_speed),
              ("Test",  test_cm,  test_metrics,  test_speed)]

    for col, (split_label, cm, m, speed) in enumerate(splits):
        ax_cm = fig.add_subplot(gs[0, col])
        im = ax_cm.imshow(cm, cmap="Blues", vmin=0)
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(class_names)
        ax_cm.set_yticklabels(class_names)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title(split_label, fontsize=13, fontweight="bold")

        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                color = "white" if cm[i, j] > total / 4 else "black"
                ax_cm.text(j, i, f"{cm[i, j]}\n({pct:.0f}%)",
                           ha="center", va="center", color=color, fontsize=13)

        fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

        ax_txt = fig.add_subplot(gs[1, col])
        ax_txt.axis("off")
        stats = (
            f"Accuracy:   {m['acc']:.3f}    Precision:  {m['prec']:.3f}\n"
            f"Recall:     {m['rec']:.3f}    F1:         {m['f1']:.3f}\n"
            f"F2:         {m['f2']:.3f}    Speed:      {speed:.1f} samp/s"
        )
        ax_txt.text(0.5, 0.5, stats, ha="center", va="center",
                    fontsize=11, family="monospace",
                    transform=ax_txt.transAxes,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4ff", edgecolor="#aabbdd"))

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix plot saved to: {save_path}")


def fmt(metrics: dict) -> str:
    return (
        f"loss={metrics['loss']:.4f}  acc={metrics['acc']:.3f}  "
        f"prec={metrics['prec']:.3f}  rec={metrics['rec']:.3f}  "
        f"f1={metrics['f1']:.3f}  f2={metrics['f2']:.3f}"
    )


def main():
    from datetime import datetime

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_dir = os.path.join("results", "lstm")
    os.makedirs(results_dir, exist_ok=True)

    embedding_label = "_".join(EMBEDDING_KEYS)
    lstm_type = "bilstm" if "Baseline" not in VideoLSTM.__name__ else "lstm"

    split_configs = [
        ("no_overlap", lambda: sample_train_test_no_overlap(
            HDF5_PATH,
            n_train_real=N_TRAIN_REAL, n_train_fake=N_TRAIN_FAKE,
            n_test_real=N_TEST_REAL,   n_test_fake=N_TEST_FAKE,
            seed=SEED, embedding_keys=EMBEDDING_KEYS,
        )),
        ("cross_dataset", lambda: sample_cross_dataset_split(
            HDF5_PATH,
            n_train_real=N_TRAIN_REAL, n_train_fake=N_TRAIN_FAKE,
            n_test_real=N_TEST_REAL,   n_test_fake=N_TEST_FAKE,
            seed=SEED, embedding_keys=EMBEDDING_KEYS,
        )),
    ]

    for split_name, get_samples in split_configs:
        print(f"\n{'#' * 80}")
        print(f"# Split: {split_name}")
        print(f"{'#' * 80}")

        # ── Data ──────────────────────────────────────────────────────────────
        print(f"\nSampling dataset ({embedding_label})...")
        train_samples, test_samples = get_samples()

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
        train_cm, train_m, train_speed = full_evaluation(model, train_loader, train_samples, device, "TRAIN")
        test_cm,  test_m,  test_speed  = full_evaluation(model, test_loader,  test_samples,  device, "TEST")

        # ── Save plots ────────────────────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_stem = f"{embedding_label}_{lstm_type}_{split_name}_{timestamp}"
        plot_title = f"{embedding_label.upper()} | {lstm_type.upper()} | {split_name.replace('_', ' ')}"

        plot_training_curve(
            losses,
            plot_title,
            os.path.join(results_dir, f"loss_{file_stem}.png"),
        )
        plot_confusion_matrices(
            train_cm, test_cm,
            train_m, test_m,
            train_speed, test_speed,
            plot_title,
            os.path.join(results_dir, f"confusion_{file_stem}.png"),
        )


if __name__ == "__main__":
    main()
