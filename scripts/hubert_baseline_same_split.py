#!/usr/bin/env python3
"""
Baseline on HuBERT-only embeddings using the same split config as combined runs.
Creates fresh result file; leaves existing files untouched.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HUBERT = ROOT / "embeddings/generated/hubert_audio_2025-08-21T13:51:04.162022+00:00.npy"
DEFAULT_LABELS = ROOT / "labels/audio/hubert/unified_hubert_labels.pkl"


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {path}")
    return np.load(path)


def load_labels(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing labels file: {path}")
    import numpy.core as np_core

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.numeric", importlib.import_module("numpy.core.numeric"))
    with path.open("rb") as f:
        arr = pickle.load(f)
    return np.asarray(arr, dtype=int)


def preload_libomp() -> Optional[str]:
    candidates = [
        ROOT / "venv/lib/python3.9/site-packages/torch/lib/libomp.dylib",
        ROOT / "venv/lib/python3.9/site-packages/sklearn/.dylibs/libomp.dylib",
        ROOT / "venv/lib/python3.9/site-packages/faiss/.dylibs/libomp.dylib",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                ctypes.cdll.LoadLibrary(str(candidate))
                return str(candidate)
            except OSError:
                continue
    return None


def to_confusion_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def run_random_forest(x_train, y_train, x_test, y_test) -> Dict[str, int]:
    clf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    return to_confusion_dict(y_test, preds)


def run_xgboost(x_train, y_train, x_test, y_test, pos_weight: float) -> Dict[str, int]:
    import xgboost as xgb

    clf = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1.0,
        gamma=0.1,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=4,
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=pos_weight,
    )
    clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
    preds = clf.predict(x_test)
    return to_confusion_dict(y_test, preds)


def main(args: argparse.Namespace) -> None:
    hubert_path = Path(args.hubert_path).resolve()
    labels_path = Path(args.labels_path).resolve()

    embeddings = load_embeddings(hubert_path)
    labels = load_labels(labels_path)
    if len(embeddings) != len(labels):
        raise ValueError("Embeddings and labels must align on first dimension.")

    pos_weight = float((labels == 0).sum() / max(1, (labels == 1).sum()))

    x_train, x_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print(f"HuBERT shape: {embeddings.shape}")
    print(f"Train size: {len(x_train)} | Test size: {len(x_test)} | pos_weight={pos_weight:.3f}")

    rf_conf = run_random_forest(x_train, y_train, x_test, y_test)
    print(f"RandomForest confusion: {rf_conf}")

    libomp_used = preload_libomp()
    try:
        xgb_conf = run_xgboost(x_train, y_train, x_test, y_test, pos_weight=pos_weight)
        print(f"XGBoost confusion: {xgb_conf}")
    except Exception as exc:  # noqa: BLE001
        xgb_conf = {"error": str(exc)}
        print("XGBoost failed:", exc)

    out_path = ROOT / "results" / "hubert_only_same_split_tree_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_by": "hubert_baseline_same_split.py",
        "embeddings": str(hubert_path.relative_to(ROOT)),
        "labels": str(labels_path.relative_to(ROOT)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "random_state": 42,
        "libomp_loaded": libomp_used,
        "pos_weight": pos_weight,
        "models": {
            "random_forest": rf_conf,
            "xgboost": xgb_conf,
        },
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    preload_libomp()
    parser = argparse.ArgumentParser(description="HuBERT-only baseline on same split.")
    parser.add_argument("--hubert-path", default=str(DEFAULT_HUBERT))
    parser.add_argument("--labels-path", default=str(DEFAULT_LABELS))
    main(parser.parse_args())
