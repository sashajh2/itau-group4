"""
Inspect per-sample predictions for a specific checkpoint group.

Loads the 5-fold checkpoints from the 20260326_144046 run, rebuilds the
identical splits, runs inference, and writes two CSV files:
  - correct_predictions.csv   (TP + TN)
  - misclassified.csv         (FP + FN)

Usage:
    python -m scripts.inspect_predictions
"""

import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformer_experiments.dataset import (
    DeepfakeSequenceDataset,
    collate_fn,
    sample_kfold_splits,
)
from transformer_experiments.model import VanillaTransformerClassifier

# ── Config ─────────────────────────────────────────────────────────────────────
HDF5_PATH = "exports/deepfake_embeddings.h5"
CHECKPOINTS = [
    "results/sequence_models/transformer/kfold_5fold_hubert_fold1_20260326_154738.pt",
    "results/sequence_models/transformer/kfold_5fold_hubert_fold2_20260326_155151.pt",
    "results/sequence_models/transformer/kfold_5fold_hubert_fold3_20260326_155606.pt",
    "results/sequence_models/transformer/kfold_5fold_hubert_fold4_20260326_160013.pt",
    "results/sequence_models/transformer/kfold_5fold_hubert_fold5_20260326_160521.pt",
]
OUT_DIR = "results/sequence_models/inspect_20260326_144046"
SEQ_LEN = 256
BATCH_SIZE = 8
# ───────────────────────────────────────────────────────────────────────────────

_LEGACY_DEFAULTS = {
    "k": 5, "n_train_per_class": 300, "n_test_per_class": 100,
    "n_shareveo3_test": 200, "dataset_filter": "avdeepfake1m", "seed": 42,
    "d_model": 256, "nhead": 8, "num_layers": 4, "dim_feedforward": 1024,
    "dropout": 0.1, "max_seq_len": 256,
}


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    for key, val in _LEGACY_DEFAULTS.items():
        ckpt.setdefault(key, val)
    model = VanillaTransformerClassifier(
        input_dim=ckpt["input_dim"],
        d_model=ckpt["d_model"],
        nhead=ckpt["nhead"],
        num_layers=ckpt["num_layers"],
        dim_feedforward=ckpt["dim_feedforward"],
        dropout=ckpt["dropout"],
        max_seq_len=ckpt["max_seq_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def infer(model, loader, device):
    all_probs, all_labels = [], []
    for batch in loader:
        logits = model(
            batch["embeddings"].to(device),
            batch["attention_mask"].to(device),
        ).squeeze(-1)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(batch["labels"].numpy())
    return np.array(all_probs), np.array(all_labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, cfg = load_checkpoint(CHECKPOINTS[0], device)
    k               = cfg["k"]
    seed            = cfg["seed"]
    embedding_keys  = cfg["embedding_keys"]
    dataset_filter  = cfg["dataset_filter"]
    n_train_per_cls = cfg["n_train_per_class"]
    n_test_per_cls  = cfg["n_test_per_class"]

    print(f"Config: k={k}  seed={seed}  embeddings={embedding_keys}")
    print(f"  dataset_filter={dataset_filter}  n_train={n_train_per_cls}  n_test={n_test_per_cls}")

    print(f"\nLoading {len(CHECKPOINTS)} models...")
    models, ckpts = [], []
    for path in CHECKPOINTS:
        m, ckpt = load_checkpoint(path, device)
        models.append(m)
        ckpts.append(ckpt)
        print(f"  Fold {ckpt['fold']}  epoch={ckpt['epoch']}  F1={ckpt['test_f1']:.3f}")

    print(f"\nRebuilding {k}-fold splits (seed={seed})...")
    splits = sample_kfold_splits(
        HDF5_PATH, k=k,
        n_train_per_class=n_train_per_cls, n_test_per_class=n_test_per_cls,
        seed=seed, embedding_keys=embedding_keys, dataset_filter=dataset_filter,
    )
    test_samples_per_fold = [ts for _, ts in splits]

    correct_rows = []
    miss_rows = []

    for fold_idx, (model, test_samples) in enumerate(zip(models, test_samples_per_fold)):
        ds = DeepfakeSequenceDataset(test_samples, max_seq_len=SEQ_LEN)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        probs, labels = infer(model, loader, device)
        preds = (probs >= 0.5).astype(int)

        for i, s in enumerate(test_samples):
            true_label = int(labels[i])
            pred_label = int(preds[i])
            outcome = (
                "TP" if true_label == 1 and pred_label == 1 else
                "TN" if true_label == 0 and pred_label == 0 else
                "FP" if true_label == 0 and pred_label == 1 else
                "FN"
            )
            row = {
                "fold": fold_idx + 1,
                "video_id": s["video_id"],
                "aug_idx": s["aug_idx"],
                "true_label": true_label,
                "true_class": "fake" if true_label == 1 else "real",
                "pred_label": pred_label,
                "pred_class": "fake" if pred_label == 1 else "real",
                "prob_fake": round(float(probs[i]), 4),
                "outcome": outcome,
            }
            if pred_label == true_label:
                correct_rows.append(row)
            else:
                miss_rows.append(row)

        n_correct = sum(1 for r in correct_rows if r["fold"] == fold_idx + 1)
        n_miss    = sum(1 for r in miss_rows    if r["fold"] == fold_idx + 1)
        print(f"  Fold {fold_idx+1}: {n_correct} correct, {n_miss} misclassified")

    os.makedirs(OUT_DIR, exist_ok=True)
    fields = ["fold", "video_id", "aug_idx", "true_label", "true_class",
              "pred_label", "pred_class", "prob_fake", "outcome"]

    correct_path = os.path.join(OUT_DIR, "correct_predictions.csv")
    miss_path    = os.path.join(OUT_DIR, "misclassified.csv")

    for path, rows in [(correct_path, correct_rows), (miss_path, miss_rows)]:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nCorrect predictions ({len(correct_rows)} rows) → {correct_path}")
    print(f"Misclassified      ({len(miss_rows)} rows) → {miss_path}")

    # Quick summary
    for outcome in ["TP", "TN", "FP", "FN"]:
        count = sum(1 for r in correct_rows + miss_rows if r["outcome"] == outcome)
        print(f"  {outcome}: {count}")


if __name__ == "__main__":
    main()
