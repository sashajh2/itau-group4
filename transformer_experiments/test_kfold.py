"""
Evaluate saved k-fold checkpoints produced by train_kfold.py.

Automatically finds the latest complete set of fold checkpoints in
RESULTS_DIR. Rebuilds the identical splits from the config stored in each
checkpoint, runs inference, and reports Accuracy, Precision, Recall, F1,
FNR, and ROC AUC for AVDeepfake and ShareVeo3 test sets.

SEQ_LENS sweeps different truncation lengths on the same saved checkpoints.
Only lengths ≤ the model's training max_seq_len are valid (positional
encoding is fixed at train time; longer sequences require retraining).

SHOW_PROBS controls per-sample probability output (primary seq_len only):
    "none"   — no per-sample scores printed
    "misses" — only mislabeled samples
    "all"    — every sample

Usage:
    python -m transformer_experiments.test_kfold
"""

import csv
import glob
import os
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from transformer_experiments.dataset import (
    DeepfakeSequenceDataset,
    collate_fn,
    sample_fixed_test,
    sample_kfold_splits,
)
from transformer_experiments.model import VanillaTransformerClassifier

# ── Config ────────────────────────────────────────────────────────────────────
HDF5_PATH = "exports/deepfake_embeddings.h5"
CHECKPOINTS_DIR = os.path.join("results", "sequence_models", "transformer")
RESULTS_DIR = os.path.join("results", "sequence_models", "seq_len_sweep")

# Set to a list of explicit paths to override auto-detection, e.g.:
#   CHECKPOINTS = ["results/sequence_models/transformer/kfold_5fold_hubert_fold1_....pt", ...]
# Leave as None to use the latest complete checkpoint group automatically.
CHECKPOINTS = None

# Truncation lengths to sweep. Must all be ≤ the model's training max_seq_len.
# The last entry is the primary length used for ROC curves + mislabeled CSV.
SEQ_LENS = [32, 64, 128, 256]

# "none" | "misses" | "all"  (printed only for the primary seq_len)
SHOW_PROBS = "misses"

BATCH_SIZE = 8
# ─────────────────────────────────────────────────────────────────────────────


# Fallback config for checkpoints saved before the config fields were added
_LEGACY_DEFAULTS = {
    "k": 5, "n_train_per_class": 300, "n_test_per_class": 100,
    "n_shareveo3_test": 200, "dataset_filter": "avdeepfake1m", "seed": 42,
    "d_model": 256, "nhead": 8, "num_layers": 4, "dim_feedforward": 1024,
    "dropout": 0.1, "max_seq_len": 256,
}


def find_latest_checkpoints(results_dir):
    """Return paths for the most recent complete set of fold checkpoints.

    Groups by run prefix (everything before _fold{n}_) so that all folds from
    the same training run are collected even though they finish at different times.
    """
    pattern = os.path.join(results_dir, "kfold_*fold_*_fold*_*.pt")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No fold checkpoints found in {results_dir}")

    # Group by run prefix: strip the trailing _fold{n}_YYYYMMDD_HHMMSS
    # e.g. "kfold_3fold_hubert_fold2_20260409_141809.pt"
    #   → prefix "kfold_3fold_hubert", date "20260409"
    groups = defaultdict(list)
    for p in paths:
        name = os.path.basename(p).replace(".pt", "")
        parts = name.split("_")
        # Find the index of the "fold{n}" token
        fold_token_idx = next(
            (i for i, t in enumerate(parts) if t.startswith("fold") and t[4:].isdigit()),
            None,
        )
        if fold_token_idx is None:
            continue
        # prefix = everything before fold{n}, date = parts[-2]
        prefix = "_".join(parts[:fold_token_idx])
        date   = parts[-2]  # YYYYMMDD
        groups[(prefix, date)].append(p)

    # Pick the group with the most recent date
    latest_key = max(groups.keys(), key=lambda k: k[1])
    return sorted(groups[latest_key])  # sorted by fold number in filename


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


def compute_metrics(probs, labels):
    preds = (probs >= 0.5).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    total = tp + fp + fn + tn

    acc  = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    fnr  = fn / max(tp + fn, 1)

    fpr_arr, tpr_arr, _ = (
        roc_curve(labels, probs) if len(np.unique(labels)) > 1
        else (np.array([0.0, 1.0]), np.array([0.0, 1.0]), None)
    )
    roc_auc = auc(fpr_arr, tpr_arr)

    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "fnr": fnr, "auc": roc_auc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fpr": fpr_arr, "tpr": tpr_arr,
    }


def print_probs(samples, probs, labels, mode):
    if mode == "none":
        return
    preds = (probs >= 0.5).astype(int)
    header = f"  {'#':>4}  {'True':>5}  {'Pred':>5}  {'P(fake)':>8}  {'':>5}  Video"
    sep = "  " + "-" * 65
    printed = False
    for i, s in enumerate(samples):
        is_miss = preds[i] != int(labels[i])
        if mode == "misses" and not is_miss:
            continue
        if not printed:
            print(header)
            print(sep)
            printed = True
        true_str = "FAKE" if labels[i] == 1 else "REAL"
        pred_str = "FAKE" if preds[i] == 1 else "REAL"
        flag = "MISS" if is_miss else "OK  "
        print(f"  {i+1:>4}  {true_str:>5}  {pred_str:>5}  {probs[i]:>8.4f}  {flag}  "
              f"{s['video_id'][:35]}  aug={s['aug_idx']}")
    if not printed:
        print("  (no samples to show)")


def make_loader(samples, seq_len):
    ds = DeepfakeSequenceDataset(samples, max_seq_len=seq_len)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


def print_metrics(m, label, indent="  "):
    print(f"{indent}{label}")
    print(f"{indent}  Acc={m['acc']:.3f}  Prec={m['prec']:.3f}  Rec={m['rec']:.3f}  "
          f"F1={m['f1']:.3f}  FNR={m['fnr']:.3f}  AUC={m['auc']:.3f}")
    print(f"{indent}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Resolve checkpoints ───────────────────────────────────────────────────
    ckpt_paths = CHECKPOINTS or find_latest_checkpoints(CHECKPOINTS_DIR)
    print(f"Using {len(ckpt_paths)} checkpoint(s):")
    for p in ckpt_paths:
        print(f"  {os.path.basename(p)}")

    # ── Load config & models once ─────────────────────────────────────────────
    _, cfg = load_checkpoint(ckpt_paths[0], device)
    k               = cfg["k"]
    seed            = cfg["seed"]
    embedding_keys  = cfg["embedding_keys"]
    dataset_filter  = cfg["dataset_filter"]
    n_train_per_cls = cfg["n_train_per_class"]
    n_test_per_cls  = cfg["n_test_per_class"]
    model_max_len   = cfg["max_seq_len"]

    for sl in SEQ_LENS:
        if sl > model_max_len:
            raise ValueError(
                f"SEQ_LEN={sl} exceeds model's training max_seq_len={model_max_len}. "
                "Retraining required for longer sequences."
            )

    print(f"\nConfig: k={k}  seed={seed}  embeddings={embedding_keys}")
    print(f"  dataset_filter={dataset_filter}  n_train_per_class={n_train_per_cls}"
          f"  n_test_per_class={n_test_per_cls}")
    print(f"  model_max_seq_len={model_max_len}  sweeping SEQ_LENS={SEQ_LENS}\n")

    print(f"Loading {len(ckpt_paths)} models...")
    models, ckpts = [], []
    for path in ckpt_paths:
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

    # ── Sweep over truncation lengths ─────────────────────────────────────────
    # sweep_results[seq_len] = {"avd": [metrics_fold0, ...]}
    sweep_results = {}
    primary_seq_len = SEQ_LENS[-1]

    for seq_len in SEQ_LENS:
        print(f"\n{'─' * 70}")
        print(f"  SEQ_LEN = {seq_len}")
        print(f"{'─' * 70}")
        avd_fold_metrics = []

        for fold_idx, (model, test_samples) in enumerate(zip(models, test_samples_per_fold)):
            test_loader = make_loader(test_samples, seq_len)
            probs_avd, labels_avd = infer(model, test_loader, device)
            m_avd = compute_metrics(probs_avd, labels_avd)
            avd_fold_metrics.append(m_avd)

            print(f"\n  Fold {fold_idx + 1}")
            print_metrics(m_avd, "AVDeepfake")

            if seq_len == primary_seq_len and SHOW_PROBS != "none":
                print(f"\n  AVDeepfake probabilities ('{SHOW_PROBS}'):")
                print_probs(test_samples, probs_avd, labels_avd, SHOW_PROBS)

        sweep_results[seq_len] = {"avd": avd_fold_metrics}

    # ── Summary table ─────────────────────────────────────────────────────────
    def mean_std(metrics_list, key):
        vals = [m[key] for m in metrics_list]
        return np.mean(vals), np.std(vals)

    print(f"\n{'=' * 90}")
    print(f"  TRUNCATION SWEEP — MEAN ± STD ACROSS {k} FOLDS")
    print(f"{'=' * 90}")

    for dataset_name, split_key in [("AVDeepfake", "avd")]:
        print(f"\n  {dataset_name}")
        cols = ["Acc", "Prec", "Rec", "F1", "FNR", "AUC"]
        keys = ["acc", "prec", "rec", "f1", "fnr", "auc"]
        header = f"  {'SeqLen':>7}  " + "  ".join(f"{c:>12}" for c in cols)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for sl in SEQ_LENS:
            ml = sweep_results[sl][split_key]
            tag = " *" if sl == primary_seq_len else "  "
            row = f"  {sl:>7}{tag} "
            for k_name in keys:
                mu, sd = mean_std(ml, k_name)
                row += f"  {mu:>5.3f}±{sd:.3f}"
            print(row)
    print(f"\n  (* = training seq_len)")

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Mislabeled CSV — primary seq_len only
    all_miss_rows = []
    for fold_idx, (model, test_samples) in enumerate(zip(models, test_samples_per_fold)):
        test_loader = make_loader(test_samples, primary_seq_len)
        probs, labels = infer(model, test_loader, device)
        preds = (probs >= 0.5).astype(int)
        for i, s in enumerate(test_samples):
            if preds[i] != int(labels[i]):
                all_miss_rows.append({
                    "fold": fold_idx + 1, "split": "avdeepfake",
                    "video_id": s["video_id"], "aug_idx": s["aug_idx"],
                    "true_label": int(labels[i]), "pred_label": int(preds[i]),
                    "prob_fake": float(probs[i]),
                })

    csv_path = os.path.join(RESULTS_DIR, f"mislabeled_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["fold", "split", "video_id", "aug_idx", "true_label", "pred_label", "prob_fake"]
        )
        writer.writeheader()
        writer.writerows(all_miss_rows)
    print(f"\nMislabeled CSV ({len(all_miss_rows)} rows, seq_len={primary_seq_len}) → {csv_path}")

    # ── Plot 1: metrics vs truncation length ──────────────────────────────────
    metrics_cfg = [
        # (split_key, metric_key, y_label, axis)
        ("avd", "acc",  "Accuracy",  (0, 0)),
        ("avd", "prec", "Precision", (0, 1)),
        ("avd", "rec",  "Recall",    (0, 2)),
        ("avd", "f1",   "F1",        (1, 0)),
        ("avd", "fnr",  "FNR",       (1, 1)),
        ("avd", "auc",  "ROC AUC",   (1, 2)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes  # shape (2, 3)

    for split_key, metric_key, ylabel, (row, col) in metrics_cfg:
        ax = axes[row, col]
        means = [np.mean([m[metric_key] for m in sweep_results[sl][split_key]]) for sl in SEQ_LENS]
        stds  = [np.std( [m[metric_key] for m in sweep_results[sl][split_key]]) for sl in SEQ_LENS]
        ax.plot(SEQ_LENS, means, marker="o")
        ax.fill_between(SEQ_LENS,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)], alpha=0.2)
        ax.axvline(primary_seq_len, color="grey", linestyle="--", linewidth=0.8,
                   label=f"train ({primary_seq_len})")
        ax.set_xlabel("Truncation length (frames)")
        ax.set_title(ylabel)
        ax.set_xticks(SEQ_LENS)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)

    fig.suptitle("Effect of Truncation Length — 5-Fold CV (hubert)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    sweep_path = os.path.join(RESULTS_DIR, f"seq_len_sweep_{timestamp}.png")
    fig.savefig(sweep_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sweep plot → {sweep_path}")

    # ── Plot 2: ROC curves for primary seq_len ────────────────────────────────
    primary = sweep_results[primary_seq_len]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    fold_aucs = [m["auc"] for m in primary["avd"]]
    for i, m in enumerate(primary["avd"]):
        ax.plot(m["fpr"], m["tpr"], alpha=0.7, label=f"Fold {i+1}  AUC={m['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"AVDeepfake (seq_len={primary_seq_len})\nmean AUC={np.mean(fold_aucs):.3f}")
    ax.legend(fontsize=8)

    fig.suptitle("ROC Curves — 5-Fold CV (hubert)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    roc_path = os.path.join(RESULTS_DIR, f"roc_kfold_{timestamp}.png")
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ROC plot → {roc_path}")


if __name__ == "__main__":
    main()
