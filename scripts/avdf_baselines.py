#!/usr/bin/env python3
"""
Baseline classifiers on AVDeepfake1M Hubert embeddings (no Sora2).

Runs:
  - MLP (simple)
  - MLP (direct_classifier-style: dims [256, 128, 64])
  - Random Forest
  - XGBoost (requires xgboost to be installed)

For the MLPs, we also compute embedding quality metrics on the last hidden layer.
For tree models we report confusion matrix counts only.

Usage (example):
  python scripts/avdf_baselines.py \
    --hdf5 data/evaluation_data/deepfake_embeddings_2.h5 \
    --encoder hubert \
    --split 0.2 \
    --output results/baselines_avdf_hubert.json
"""
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from training.disentangled.metrics import (
    compute_clustering_metrics,
    compute_distribution_metrics,
    compute_separation_metrics,
)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_avdf_hubert(
    hdf5_path: str,
    encoder: str = "hubert",
    max_samples: int = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load AV1M/AVDeepfake subset only (filter by path containing 'av1m')."""
    embeddings = []
    labels = []

    with h5py.File(hdf5_path, "r") as f:
        videos_group = f["/videos"]
        total_videos = len(videos_group.keys())
        for idx, vid in enumerate(videos_group.keys()):
            if max_samples and len(embeddings) >= max_samples:
                break
            video = videos_group[vid]
            if "augmentation_info" not in video:
                continue
            aug_info = video["augmentation_info"]
            paths = aug_info["video_paths"][:]
            if len(paths) == 0:
                continue
            video_path = paths[0].decode().lower()
            if "av1m" not in video_path and "avdeepfake" not in video_path:
                continue  # exclude ShareVeo and other sources

            if f"embeddings/{encoder}" not in video:
                continue

            emb = video[f"embeddings/{encoder}"][:]  # [num_augs, num_segs, dim]
            audio_labels = video["labels/audio"][:]
            video_labels = video["labels/video"][:]
            num_augs, num_segs, _ = emb.shape

            for aug_idx in range(num_augs):
                for seg_idx in range(num_segs):
                    if max_samples and len(embeddings) >= max_samples:
                        break
                    is_real = (audio_labels[aug_idx, seg_idx] == 0) and (
                        video_labels[aug_idx, seg_idx] == 0
                    )
                    label = 0 if is_real else 1
                    embeddings.append(emb[aug_idx, seg_idx])
                    labels.append(label)
            if verbose and (idx + 1) % 200 == 0:
                print(
                    f"  Processed {idx+1}/{total_videos} videos | "
                    f"samples loaded: {len(embeddings)}"
                )

    embeddings = np.array(embeddings)
    labels = np.array(labels, dtype=int)
    return embeddings, labels


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        logits = self.fc_out(h)
        return logits, h


class SashaMLP(nn.Module):
    """Direct-classifier-style MLP similar to models/configs/direct_classifier.json."""

    def __init__(self, input_dim: int, dims=(256, 128, 64)):
        super().__init__()
        layers = []
        last = input_dim
        for d in dims:
            layers.append(nn.Linear(last, d))
            layers.append(nn.ReLU())
            last = d
        self.hidden = nn.Sequential(*layers)
        self.fc_out = nn.Linear(last, 1)

    def forward(self, x):
        h = self.hidden(x)
        logits = self.fc_out(h)
        return logits, h


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int):
    tensor_x = torch.from_numpy(x).float()
    tensor_y = torch.from_numpy(y).float().unsqueeze(1)
    ds = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


def train_mlp(model: nn.Module, x_train, y_train, x_val, y_val, device, cfg: TrainConfig):
    train_loader = make_loader(x_train, y_train, cfg.batch_size)
    val_loader = make_loader(x_val, y_val, cfg.batch_size)

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    pos_weight = torch.tensor(neg / max(1, pos), device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.to(device)
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        steps = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            steps += 1
        avg_loss = running / max(1, steps)
        print(f"    Epoch {epoch+1}/{cfg.epochs} loss={avg_loss:.4f}")

    # Eval
    model.eval()
    with torch.no_grad():
        logits_list = []
        hidden_list = []
        y_true = []
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits, h = model(xb)
            logits_list.append(logits.cpu())
            hidden_list.append(h.cpu())
            y_true.append(yb)

    logits = torch.cat(logits_list).squeeze(1)
    hidden = torch.cat(hidden_list).numpy()
    y_true = torch.cat(y_true).squeeze(1).numpy().astype(int)
    y_prob = torch.sigmoid(logits).numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }

    # Embedding quality on hidden layer
    emb_metrics: Dict[str, float] = {}
    emb_metrics.update(compute_clustering_metrics(hidden, y_true, metric="cosine"))
    emb_metrics.update(compute_distribution_metrics(hidden, y_true))
    emb_metrics.update(compute_separation_metrics(hidden, y_true))

    return metrics, emb_metrics


def run_random_forest(x_train, y_train, x_val, y_val):
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
    return {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}


def run_xgboost(x_train, y_train, x_val, y_val):
    try:
        import xgboost as xgb
    except ImportError:
        return {"error": "xgboost not installed"}

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / max(1, pos)
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    clf.fit(x_train, y_train)
    y_pred = (clf.predict_proba(x_val)[:, 1] >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
    return {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AVDF Hubert baselines (no Sora2)")
    parser.add_argument("--hdf5", type=str, required=True, help="Path to HDF5")
    parser.add_argument("--encoder", type=str, default="hubert", help="Encoder name")
    parser.add_argument("--split", type=float, default=0.2, help="Val split fraction")
    parser.add_argument("--output", type=str, default="results/baselines_avdf_hubert.json")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of samples")
    parser.add_argument("--verbose", action="store_true", help="Print progress while loading")
    args = parser.parse_args()

    print("📂 Loading AVDeepfake/AV1M subset...")
    X, y = load_avdf_hubert(
        args.hdf5,
        encoder=args.encoder,
        max_samples=args.max_samples,
        verbose=args.verbose,
    )
    print(f"✅ Loaded {len(X):,} samples from AV1M subset. Real={np.sum(y==0):,}, Fake={np.sum(y==1):,}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.split, stratify=y, random_state=42
    )

    # Standardize for MLPs (not strictly needed for trees)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainConfig()

    results = {
        "meta": {
            "hdf5": args.hdf5,
            "encoder": args.encoder,
            "val_split": args.split,
            "num_samples": len(X),
            "num_train": len(X_train),
            "num_val": len(X_val),
            "real_train": int(np.sum(y_train == 0)),
            "fake_train": int(np.sum(y_train == 1)),
            "real_val": int(np.sum(y_val == 0)),
            "fake_val": int(np.sum(y_val == 1)),
        }
    }

    # MLP (simple)
    print("\n▶️ Training mlp_simple...")
    mlp_simple = SimpleMLP(input_dim=X.shape[1])
    m_metrics, m_emb = train_mlp(mlp_simple, X_train_std, y_train, X_val_std, y_val, device, cfg)
    results["mlp_simple"] = {"confusion": m_metrics, "embedding_metrics": m_emb}
    print(f"   Confusion: {m_metrics}")

    # MLP (Sasha / direct-classifier style)
    print("\n▶️ Training mlp_sasha (direct-classifier style)...")
    mlp_sasha = SashaMLP(input_dim=X.shape[1])
    s_metrics, s_emb = train_mlp(mlp_sasha, X_train_std, y_train, X_val_std, y_val, device, cfg)
    results["mlp_sasha"] = {"confusion": s_metrics, "embedding_metrics": s_emb}
    print(f"   Confusion: {s_metrics}")

    # Random Forest
    print("\n▶️ Training random_forest...")
    rf_metrics = run_random_forest(X_train, y_train, X_val, y_val)
    results["random_forest"] = {"confusion": rf_metrics}
    print(f"   Confusion: {rf_metrics}")

    # XGBoost
    print("\n▶️ Training xgboost...")
    xgb_metrics = run_xgboost(X_train, y_train, X_val, y_val)
    results["xgboost"] = {"confusion": xgb_metrics}
    print(f"   Confusion: {xgb_metrics}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
