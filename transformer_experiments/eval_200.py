"""
Run a single evaluation on 200 AVDeepfake samples (100 real + 100 fake)
using the latest saved checkpoint, and record every video ID, true label,
predicted label, and probability score to a CSV.

Usage:
    python -m transformer_experiments.eval_200
"""

import csv
import glob
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformer_experiments.dataset import (
    DeepfakeSequenceDataset,
    collate_fn,
    sample_kfold_splits,
)
from transformer_experiments.model import VanillaTransformerClassifier

# ── Config ────────────────────────────────────────────────────────────────────
HDF5_PATH       = "exports/deepfake_embeddings.h5"
CHECKPOINTS_DIR = os.path.join("results", "transformer")

# Set to a specific .pt path to override auto-detection, e.g.:
#   CHECKPOINT_OVERRIDE = "results/transformer/kfold_5fold_hubert_fold5_20260326_160521.pt"
# Leave as None to use the most recently modified checkpoint.
CHECKPOINT_OVERRIDE = "results/transformer/kfold_5fold_hubert_fold5_20260326_160521.pt"
RESULTS_DIR     = os.path.join("results", "eval_200")

N_SAMPLES_PER_CLASS = 100   # 100 real + 100 fake = 200 total
BATCH_SIZE          = 8
SEED                = 99    # different seed from training so samples are independent
# ─────────────────────────────────────────────────────────────────────────────

_LEGACY_DEFAULTS = {
    "k": 5, "n_train_per_class": 300, "n_test_per_class": 100,
    "dataset_filter": "avdeepfake1m", "seed": 42,
    "d_model": 256, "nhead": 8, "num_layers": 4, "dim_feedforward": 1024,
    "dropout": 0.1, "max_seq_len": 256,
}


def find_latest_checkpoint(results_dir):
    """Return the single most recently modified checkpoint."""
    paths = glob.glob(os.path.join(results_dir, "kfold_*fold_*_fold*_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {results_dir}")
    return max(paths, key=os.path.getmtime)


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    ckpt_path = CHECKPOINT_OVERRIDE or find_latest_checkpoint(CHECKPOINTS_DIR)
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    model, cfg = load_checkpoint(ckpt_path, device)
    print(f"  Fold {cfg['fold']}  epoch={cfg['epoch']}  F1={cfg['test_f1']:.3f}")

    embedding_keys  = cfg["embedding_keys"]
    dataset_filter  = cfg["dataset_filter"]
    max_seq_len     = cfg["max_seq_len"]

    # Sample 200 fresh AVDeepfake videos using a separate seed
    # Use k=1 split so all available videos are in the "train" pool, then take the test slice
    print(f"\nSampling {N_SAMPLES_PER_CLASS * 2} AVDeepfake videos (seed={SEED})...")
    splits = sample_kfold_splits(
        HDF5_PATH, k=2,
        n_train_per_class=N_SAMPLES_PER_CLASS,
        n_test_per_class=N_SAMPLES_PER_CLASS,
        seed=SEED,
        embedding_keys=embedding_keys,
        dataset_filter=dataset_filter,
    )
    # Take the test set from fold 1
    samples = splits[0][1]
    print(f"  Got {len(samples)} samples  "
          f"({sum(s['label']==0 for s in samples)} real, {sum(s['label']==1 for s in samples)} fake)")

    # Run inference
    ds     = DeepfakeSequenceDataset(samples, max_seq_len=max_seq_len)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["embeddings"].to(device),
                batch["attention_mask"].to(device),
            ).squeeze(-1)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= 0.5).astype(int)

    # Metrics
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    acc  = (tp + tn) / len(labels)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    fnr  = fn / max(tp + fn, 1)

    print(f"\nResults ({len(samples)} samples):")
    print(f"  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  FNR={fnr:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # Save all predictions to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = os.path.join(RESULTS_DIR, f"predictions_200_{timestamp}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_id", "aug_idx", "true_label", "pred_label", "prob_fake", "correct"
        ])
        writer.writeheader()
        for i, s in enumerate(samples):
            writer.writerow({
                "video_id":   s["video_id"],
                "aug_idx":    s["aug_idx"],
                "true_label": int(labels[i]),
                "pred_label": int(preds[i]),
                "prob_fake":  f"{probs[i]:.4f}",
                "correct":    int(preds[i] == int(labels[i])),
            })

    print(f"\nAll {len(samples)} predictions saved → {csv_path}")

    # Print missed video IDs
    misses = [(s["video_id"], s["aug_idx"], int(labels[i]), int(preds[i]), probs[i])
              for i, s in enumerate(samples) if preds[i] != int(labels[i])]
    print(f"\nMissed ({len(misses)} / {len(samples)}):")
    print(f"  {'video_id':<35}  {'aug':>3}  {'true':>5}  {'pred':>5}  {'P(fake)':>8}")
    print(f"  {'-' * 65}")
    for vid, aug, true, pred, prob in misses:
        true_str = "FAKE" if true == 1 else "REAL"
        pred_str = "FAKE" if pred == 1 else "REAL"
        print(f"  {vid:<35}  {aug:>3}  {true_str:>5}  {pred_str:>5}  {prob:>8.4f}")


if __name__ == "__main__":
    main()
