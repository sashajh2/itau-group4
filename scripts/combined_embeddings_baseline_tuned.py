#!/usr/bin/env python3
"""
Tuned variant of combined_embeddings_baseline:
  - Same concatenation of OpenL3 + HuBERT + SeNet
  - More aggressive RandomForest/XGBoost hyperparameters
  - Early stopping for XGBoost

Writes fresh combined embeddings/mapping/results; does not touch existing files.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
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
DEFAULT_OPENL3 = ROOT / "embeddings/generated/openl3_audio_2025-08-21T13:51:04.162022+00:00.npy"
DEFAULT_HUBERT = ROOT / "embeddings/generated/hubert_audio_2025-08-21T13:51:04.162022+00:00.npy"
DEFAULT_SENET = ROOT / "embeddings/generated/senet_video_2025-08-21T13:51:04.162022+00:00.npy"
DEFAULT_MAPPING = ROOT / "embeddings/generated/openl3_audio_2025-08-21T13:51:04.162022+00:00_mapping.json"
DEFAULT_LABELS = ROOT / "labels/audio/hubert/unified_hubert_labels.pkl"


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {path}")
    return np.load(path)


def load_mapping(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing mapping file: {path}")
    with path.open() as f:
        return json.load(f)


def load_labels(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing labels file: {path}")
    # Shim for older numpy pickle names.
    import numpy.core as np_core

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.numeric", importlib.import_module("numpy.core.numeric"))
    with path.open("rb") as f:
        arr = pickle.load(f)
    return np.asarray(arr, dtype=int)


def preload_libomp() -> Optional[str]:
    """Load a bundled libomp so xgboost can import on macOS without brew."""
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
        min_samples_split=2,
        min_samples_leaf=1,
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
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    openl3_path = Path(args.openl3_path).resolve()
    hubert_path = Path(args.hubert_path).resolve()
    senet_path = Path(args.senet_path).resolve()
    mapping_path = Path(args.mapping_path).resolve()
    labels_path = Path(args.labels_path).resolve()

    openl3 = load_embeddings(openl3_path)
    hubert = load_embeddings(hubert_path)
    senet = load_embeddings(senet_path)
    labels = load_labels(labels_path)
    mapping = load_mapping(mapping_path)

    if not (len(openl3) == len(hubert) == len(senet) == len(labels)):
        raise ValueError("Embeddings and labels must have the same first dimension.")

    combined = np.concatenate([openl3, hubert, senet], axis=1)
    pos_weight = float((labels == 0).sum() / max(1, (labels == 1).sum()))

    combo_name = f"combined_openl3_hubert_senet_tuned_{timestamp}"
    combined_path = ROOT / "embeddings" / "generated" / f"{combo_name}.npy"
    combined_mapping_path = ROOT / "embeddings" / "generated" / f"{combo_name}_mapping.json"
    results_path = ROOT / "results" / f"{combo_name}_tree_results.json"

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(combined_path, combined)

    combined_mapping = dict(mapping)
    combined_mapping.update(
        {
            "model": "openl3+hubert+senet (tuned trees)",
            "embedding_dimension": int(combined.shape[1]),
            "created_at": timestamp,
            "source_files": {
                "openl3": str(openl3_path.relative_to(ROOT)),
                "hubert": str(hubert_path.relative_to(ROOT)),
                "senet": str(senet_path.relative_to(ROOT)),
                "labels": str(labels_path.relative_to(ROOT)),
            },
            "notes": "Concatenated [openl3 | hubert | senet]. Tuned RF/XGB.",
        }
    )
    with combined_mapping_path.open("w") as f:
        json.dump(combined_mapping, f, indent=2)

    x_train, x_test, y_train, y_test = train_test_split(
        combined,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print(f"Combined shape: {combined.shape}")
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

    results = {
        "created_at": timestamp,
        "combined_embeddings": str(combined_path.relative_to(ROOT)),
        "combined_mapping": str(combined_mapping_path.relative_to(ROOT)),
        "label_file": str(labels_path.relative_to(ROOT)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "random_state": 42,
        "libomp_loaded": libomp_used,
        "models": {
            "random_forest": rf_conf,
            "xgboost": xgb_conf,
        },
        "pos_weight": pos_weight,
    }
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(
        f"\nSaved combined embeddings to: {combined_path}\n"
        f"Saved mapping to:           {combined_mapping_path}\n"
        f"Saved results to:           {results_path}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuned combined embeddings baselines.")
    parser.add_argument("--openl3-path", default=str(DEFAULT_OPENL3))
    parser.add_argument("--hubert-path", default=str(DEFAULT_HUBERT))
    parser.add_argument("--senet-path", default=str(DEFAULT_SENET))
    parser.add_argument("--mapping-path", default=str(DEFAULT_MAPPING))
    parser.add_argument("--labels-path", default=str(DEFAULT_LABELS))
    main(parser.parse_args())
