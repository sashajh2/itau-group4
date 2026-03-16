"""
Cross-dataset training script for the vanilla transformer deepfake detector.

Train on avdeepfake1m (real + fake), test on:
  - Real: held-out avdeepfake1m real videos
  - Fake: shareveo3 videos (completely unseen dataset)

Usage:
    python -m transformer_experiments.train_cross_dataset
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
    sample_cross_dataset_split,
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

# ── Config ────────────────────────────────────────────────────────────────────
HDF5_PATH = "exports/deepfake_embeddings.h5"

TRAIN_DATASET     = "avdeepfake1m"
TEST_FAKE_DATASET = "shareveo3"

N_TRAIN_REAL = 200
N_TRAIN_FAKE = 200
N_TEST_REAL  = 100
N_TEST_FAKE  = 100

SEED = 42

# Embeddings to use — concatenated along the feature dim.
# Available: "hubert" (768), "openl3" (512), "senet" (2048)
EMBEDDING_KEYS = ("hubert",)
EMBEDDING_DIMS  = {"hubert": 768, "openl3": 512, "senet": 2048}
INPUT_DIM = sum(EMBEDDING_DIMS[k] for k in EMBEDDING_KEYS)

EPOCHS     = 10
BATCH_SIZE = 8
LR         = 1e-4
WEIGHT_DECAY = 1e-4

MAX_SEQ_LEN = 256

D_MODEL    = 256
NHEAD      = 8
NUM_LAYERS = 4
DIM_FF     = 1024
DROPOUT    = 0.1
# ─────────────────────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    print(f"\nCross-dataset split:")
    print(f"  Train: {TRAIN_DATASET}  |  Test fake: {TEST_FAKE_DATASET}")
    print(f"Embeddings: {' + '.join(EMBEDDING_KEYS)}  (input_dim={INPUT_DIM})")

    train_samples, test_samples = sample_cross_dataset_split(
        HDF5_PATH,
        n_train_real=N_TRAIN_REAL,
        n_train_fake=N_TRAIN_FAKE,
        n_test_real=N_TEST_REAL,
        n_test_fake=N_TEST_FAKE,
        train_dataset=TRAIN_DATASET,
        test_fake_dataset=TEST_FAKE_DATASET,
        seed=SEED,
        embedding_keys=EMBEDDING_KEYS,
    )

    all_samples = train_samples + test_samples
    n_truncated = sum(1 for s in all_samples if s["embeddings"].shape[0] > MAX_SEQ_LEN)
    print(f"Padding/truncating all sequences to T={MAX_SEQ_LEN} ({n_truncated} truncated)")

    train_ds = DeepfakeSequenceDataset(train_samples, max_seq_len=MAX_SEQ_LEN)
    test_ds  = DeepfakeSequenceDataset(test_samples,  max_seq_len=MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    n_train_real = sum(1 for s in train_samples if s["label"] == 0)
    n_train_fake = sum(1 for s in train_samples if s["label"] == 1)
    n_test_real  = sum(1 for s in test_samples  if s["label"] == 0)
    n_test_fake  = sum(1 for s in test_samples  if s["label"] == 1)
    print(f"Train: {len(train_samples)} ({n_train_real} real [{TRAIN_DATASET}], {n_train_fake} fake [{TRAIN_DATASET}])")
    print(f"Test:  {len(test_samples)} ({n_test_real} real [{TRAIN_DATASET}], {n_test_fake} fake [{TEST_FAKE_DATASET}])")

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
    best_epoch   = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics  = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0

        tag = ""
        if test_metrics["f1"] > best_test_f1:
            best_test_f1 = test_metrics["f1"]
            best_epoch   = epoch
            tag = " *"

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  ({elapsed:.1f}s)  "
            f"train: {fmt(train_metrics)}  |  test: {fmt(test_metrics)}{tag}"
        )

    print("-" * 80)
    print(f"Best test F1: {best_test_f1:.3f} (epoch {best_epoch})")

    # ── Final Evaluation ──────────────────────────────────────────────────
    train_cm = full_evaluation(model, train_loader, train_samples, device, "TRAIN")
    test_cm  = full_evaluation(model, test_loader,  test_samples,  device, "TEST")

    # ── Save ──────────────────────────────────────────────────────────────
    results_dir = os.path.join("results", "transformer")
    os.makedirs(results_dir, exist_ok=True)
    emb_tag   = "_".join(EMBEDDING_KEYS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(
        results_dir,
        f"confusion_matrices_cross_{TRAIN_DATASET}_vs_{TEST_FAKE_DATASET}_{emb_tag}_{timestamp}.png",
    )
    plot_confusion_matrices(train_cm, test_cm, plot_path)


if __name__ == "__main__":
    main()
