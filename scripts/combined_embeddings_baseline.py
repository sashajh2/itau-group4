#!/usr/bin/env python3
"""
Combine OpenL3, HuBERT, and SeNet segment embeddings into a single array,
persist the combined embeddings/mapping, and run tree baselines
(RandomForest and XGBoost) on the full set.

Outputs:
  - embeddings/generated/combined_openl3_hubert_senet_<timestamp>.npy
  - embeddings/generated/combined_openl3_hubert_senet_<timestamp>_mapping.json
  - results/combined_tree_baselines_<timestamp>.json

Existing files are left untouched.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import pickle
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
    # Older pickles reference numpy._core.*, which newer NumPy versions no longer expose.
    import importlib

    import numpy.core as np_core

    sys_modules = __import__("sys").modules
    sys_modules.setdefault("numpy._core", np_core)
    sys_modules.setdefault("numpy._core.numeric", importlib.import_module("numpy.core.numeric"))
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
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    return to_confusion_dict(y_test, preds)


def run_xgboost(x_train, y_train, x_test, y_test) -> Dict[str, int]:
    # Defer import until after libomp preload to avoid import errors.
    import xgboost as xgb

    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
        eval_metric="logloss",
        tree_method="hist",
    )
    clf.fit(x_train, y_train)
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

    combo_name = f"combined_openl3_hubert_senet_{timestamp}"
    combined_path = ROOT / "embeddings" / "generated" / f"{combo_name}.npy"
    combined_mapping_path = ROOT / "embeddings" / "generated" / f"{combo_name}_mapping.json"
    results_path = (
        Path(args.results_path).resolve()
        if args.results_path
        else ROOT / "results" / f"{combo_name}_tree_results.json"
    )

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(combined_path, combined)

    combined_mapping = dict(mapping)
    combined_mapping.update(
        {
            "model": "openl3+hubert+senet",
            "embedding_dimension": int(combined.shape[1]),
            "created_at": timestamp,
            "source_files": {
                "openl3": str(openl3_path.relative_to(ROOT)),
                "hubert": str(hubert_path.relative_to(ROOT)),
                "senet": str(senet_path.relative_to(ROOT)),
                "labels": str(labels_path.relative_to(ROOT)),
            },
            "notes": "Concatenated along feature dimension: [openl3 | hubert | senet].",
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
    print(f"Train size: {len(x_train)} | Test size: {len(x_test)}")

    rf_conf = run_random_forest(x_train, y_train, x_test, y_test)
    print(f"RandomForest confusion: {rf_conf}")

    xgb_conf: Dict[str, int] | Dict[str, str]
    libomp_used = preload_libomp()
    try:
        xgb_conf = run_xgboost(x_train, y_train, x_test, y_test)
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
    }
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved combined embeddings to: {combined_path}")
    print(f"Saved mapping to:           {combined_mapping_path}")
    print(f"Saved results to:           {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine embeddings and run tree baselines.")
    parser.add_argument("--openl3-path", default=str(DEFAULT_OPENL3), help="Path to OpenL3 npy.")
    parser.add_argument("--hubert-path", default=str(DEFAULT_HUBERT), help="Path to HuBERT npy.")
    parser.add_argument("--senet-path", default=str(DEFAULT_SENET), help="Path to SeNet npy.")
    parser.add_argument(
        "--mapping-path", default=str(DEFAULT_MAPPING), help="Path to mapping json (for segments)."
    )
    parser.add_argument("--labels-path", default=str(DEFAULT_LABELS), help="Path to labels pickle.")
    parser.add_argument(
        "--results-path",
        default=None,
        help="Optional custom path for results json (directories auto-created).",
    )
    main(parser.parse_args())
