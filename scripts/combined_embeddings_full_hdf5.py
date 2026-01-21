#!/usr/bin/env python3
"""
Combine OpenL3, HuBERT, and SeNet embeddings from the full AVDF/AV1M HDF5
(`data/evaluation_data/deepfake_embeddings_2.h5`) and run tree baselines.

Outputs (timestamped):
  - embeddings/generated/combined_openl3_hubert_senet_full_<ts>.npy
  - embeddings/generated/combined_openl3_hubert_senet_full_<ts>_mapping.json
  - results/baseline_results/combined_openl3_hubert_senet_full_<ts>_tree_results.json

Existing files are not modified.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HDF5 = ROOT / "data/evaluation_data/deepfake_embeddings_2.h5"
TMP_MEMMAP = ROOT / "combined_full_tmp.dat"


# --------------------------------------------------------------------------- #
# Utils
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def count_samples(hdf5_path: Path) -> int:
    total = 0
    with h5py.File(hdf5_path, "r") as f:
        for vid_key in f["/videos"].keys():
            video = f["/videos"][vid_key]
            if "augmentation_info" not in video:
                continue
            paths = video["augmentation_info"]["video_paths"][:]
            if len(paths) == 0:
                continue
            video_path = paths[0].decode().lower()
            if "av1m" not in video_path and "avdeepfake" not in video_path:
                continue
            if not all(f"embeddings/{enc}" in video for enc in ("hubert", "openl3", "senet")):
                continue
            hubert = video["embeddings/hubert"]
            num_augs, num_segs, _ = hubert.shape
            total += num_augs * num_segs
    return total


def load_and_combine_embeddings(hdf5_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Iterate the HDF5 and concatenate [openl3 | hubert | senet] per segment."""
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Missing HDF5 file: {hdf5_path}")

    print("Counting samples...", flush=True)
    total_samples = count_samples(hdf5_path)
    if total_samples == 0:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=int), []

    combined = np.memmap(
        TMP_MEMMAP,
        dtype=np.float32,
        mode="w+",
        shape=(total_samples, 512 + 768 + 2048),
    )
    labels = np.empty((total_samples,), dtype=int)
    segment_ids: List[str] = ["" for _ in range(total_samples)]

    write_ptr = 0
    with h5py.File(hdf5_path, "r") as f:
        videos = f["/videos"]
        total_videos = len(videos.keys())
        for idx, vid_key in enumerate(videos.keys()):
            video = videos[vid_key]
            if "augmentation_info" not in video:
                continue
            aug_info = video["augmentation_info"]
            paths = aug_info["video_paths"][:]
            if len(paths) == 0:
                continue
            video_path = paths[0].decode().lower()
            if "av1m" not in video_path and "avdeepfake" not in video_path:
                continue  # exclude ShareVeo and other sources

            # Ensure required embeddings exist
            if not all(f"embeddings/{enc}" in video for enc in ("hubert", "openl3", "senet")):
                continue

            hubert = video["embeddings/hubert"][:]  # [num_augs, num_segs, 768]
            openl3 = video["embeddings/openl3"][:]  # [num_augs, num_segs, 512]
            senet = video["embeddings/senet"][:]  # [num_augs, num_segs, 2048]
            audio_labels = video["labels/audio"][:]
            video_labels = video["labels/video"][:]
            seg_ids_ds = video.get("segment_ids", None)

            num_augs, num_segs, _ = hubert.shape
            flat_hubert = hubert.reshape(-1, hubert.shape[-1])
            flat_openl3 = openl3.reshape(-1, openl3.shape[-1])
            flat_senet = senet.reshape(-1, senet.shape[-1])
            flat_labels = (
                ((audio_labels.reshape(-1) != 0) | (video_labels.reshape(-1) != 0))
                .astype(int)
                .tolist()
            )
            combined_block = np.concatenate([flat_openl3, flat_hubert, flat_senet], axis=1)

            block_len = combined_block.shape[0]
            combined[write_ptr : write_ptr + block_len] = combined_block
            labels[write_ptr : write_ptr + block_len] = flat_labels

            # Build segment identifiers aligned to flattened order.
            if seg_ids_ds is not None and len(seg_ids_ds.shape) == 1 and len(seg_ids_ds) == num_segs:
                seg_strs = [seg_ids_ds[i].decode() for i in range(num_segs)]
            else:
                seg_strs = ["" for _ in range(num_segs)]
            for aug_idx in range(num_augs):
                for seg_idx in range(num_segs):
                    global_idx = write_ptr + aug_idx * num_segs + seg_idx
                    seg_id = seg_strs[seg_idx] if seg_idx < len(seg_strs) else ""
                    segment_ids[global_idx] = f"{video_path}|aug{aug_idx}|seg{seg_idx}|{seg_id}"

            write_ptr += block_len

            if (idx + 1) % 200 == 0:
                print(
                    f"  Processed {idx+1}/{total_videos} videos | "
                    f"samples written: {write_ptr}/{total_samples}",
                    flush=True,
                )

    # Trim in case of mismatch (should not happen but safe).
    combined = combined[:write_ptr]
    labels = labels[:write_ptr]
    segment_ids = segment_ids[:write_ptr]
    return combined, labels, segment_ids


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def run_random_forest(x_train, y_train, x_test, y_test) -> Dict[str, int]:
    # Lighter settings to finish within reasonable time on full dataset.
    clf = RandomForestClassifier(
        n_estimators=100,
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
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args: argparse.Namespace) -> None:
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    hdf5_path = Path(args.hdf5_path).resolve()

    print(f"Loading and combining embeddings from {hdf5_path} ...", flush=True)
    combined, labels, segment_ids = load_and_combine_embeddings(hdf5_path)

    if combined.size == 0:
        raise RuntimeError("No samples loaded. Check filters (av1m/avdeepfake) and embeddings presence.")

    pos_weight = float((labels == 0).sum() / max(1, (labels == 1).sum()))

    combo_name = f"combined_openl3_hubert_senet_full_{timestamp}"
    combined_path = ROOT / "embeddings" / "generated" / f"{combo_name}.npy"
    combined_mapping_path = ROOT / "embeddings" / "generated" / f"{combo_name}_mapping.json"
    results_path = ROOT / "results" / "baseline_results" / f"{combo_name}_tree_results.json"

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(combined_path, combined)
    try:
        TMP_MEMMAP.unlink(missing_ok=True)
    except Exception:
        pass

    mapping = {
        "model": "openl3+hubert+senet (full hdf5)",
        "mode": "audio+video",
        "created_at": timestamp,
        "total_embeddings": int(len(combined)),
        "embedding_dimension": int(combined.shape[1]),
        "segment_ids": segment_ids,
        "source_hdf5": str(hdf5_path.relative_to(ROOT)),
        "notes": "Concatenated [openl3 | hubert | senet] per segment; filtered av1m/avdeepfake videos.",
    }
    with combined_mapping_path.open("w") as f:
        json.dump(mapping, f, indent=2)

    x_train, x_test, y_train, y_test = train_test_split(
        combined,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print(
        f"Combined shape: {combined.shape} | Train: {len(x_train)} | Test: {len(x_test)} | "
        f"pos_weight={pos_weight:.3f}",
        flush=True,
    )

    rf_conf = run_random_forest(x_train, y_train, x_test, y_test)
    print(f"RandomForest confusion: {rf_conf}", flush=True)

    libomp_used = preload_libomp()
    try:
        xgb_conf = run_xgboost(x_train, y_train, x_test, y_test, pos_weight=pos_weight)
        print(f"XGBoost confusion: {xgb_conf}", flush=True)
    except Exception as exc:  # noqa: BLE001
        xgb_conf = {"error": str(exc)}
        print("XGBoost failed:", exc, flush=True)

    results = {
        "created_at": timestamp,
        "combined_embeddings": str(combined_path.relative_to(ROOT)),
        "combined_mapping": str(combined_mapping_path.relative_to(ROOT)),
        "hdf5": str(hdf5_path.relative_to(ROOT)),
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
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved combined embeddings to: {combined_path}")
    print(f"Saved mapping to:           {combined_mapping_path}")
    print(f"Saved results to:           {results_path}", flush=True)


if __name__ == "__main__":
    preload_libomp()
    parser = argparse.ArgumentParser(description="Combine OpenL3+HuBERT+SeNet from full HDF5 and run tree baselines.")
    parser.add_argument("--hdf5-path", default=str(DEFAULT_HDF5))
    main(parser.parse_args())
