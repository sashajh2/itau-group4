import numpy as np  # type: ignore
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from collections import defaultdict
import json
from typing import List, Dict, Any, Optional
import os

def evaluate_model(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metadata: Optional[List[Dict[str, Any]]] = None,
    display_title: Optional[str] = None,
    stratify_by: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate model performance globally and, if specified, stratified by metadata fields.
    """
    results = {}
    # --- Global metrics ---
    results['global'] = _evaluate_metrics(y_true, y_probs, display_title)

    # --- Stratified metrics ---
    if metadata is not None and stratify_by:
        results['stratified'] = {}
        for field in stratify_by:
            field_groups = defaultdict(list)
            for idx, meta in enumerate(metadata):
                field_groups[meta.get(field, 'unknown')].append(idx)
            for value, indices in field_groups.items():
                y_true_g = np.array(y_true)[indices]
                y_probs_g = np.array(y_probs)[indices]
                if len(np.unique(y_true_g)) < 2:
                    continue  # Skip if only one class present
                group_title = f"{display_title or ''} [{field}={value}]"
                results['stratified'][(field, value)] = _evaluate_metrics(y_true_g, y_probs_g, group_title)
    return results


def _evaluate_metrics(y_true, y_probs, display_title=None) -> Dict[str, Any]:
    """
    Compute AUROC, FNR@1%FPR, accuracy, and plot confusion matrices.
    """
    results = {}
    results['roc_auc'] = roc_auc_score(y_true, y_probs)

    # FNR at FPR <= 1%
    best_fnr = None
    best_thresh_fpr = None
    best_cm_fpr = None
    lowest_fpr = 1.0
    fallback = None

    for t in np.arange(0.01, 1.0, 0.01):
        preds = (y_probs > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        if fpr <= 0.01 and (best_fnr is None or fnr < best_fnr):
            best_fnr = fnr
            best_thresh_fpr = t
            best_cm_fpr = confusion_matrix(y_true, preds, labels=[0, 1])
        if fpr < lowest_fpr:
            lowest_fpr = fpr
            fallback = {
                'fpr': fpr,
                'fnr': fnr,
                'threshold': t,
                'cm': confusion_matrix(y_true, preds, labels=[0, 1]),
            }
    if best_thresh_fpr is None and fallback:
        best_thresh_fpr = fallback['threshold']
        best_fnr = fallback['fnr']
        best_cm_fpr = fallback['cm']
        results['note'] = f"FPR > 1% at all thresholds — fallback used with FPR={fallback['fpr']:.3f}"
    results['best_thresh_fpr'] = best_thresh_fpr
    results['fnr_at_1%fpr'] = best_fnr
    if best_cm_fpr is not None:
        ConfusionMatrixDisplay(best_cm_fpr, display_labels=["Real", "Fake"]).plot(cmap='Blues')
        plt.title(f"{display_title or ''} Threshold={best_thresh_fpr:.2f} (FPR ≤ 1%)")
        plt.show()

    # Best threshold for accuracy
    best_acc = 0
    best_thresh_acc = None
    best_cm_acc = None
    for t in np.arange(0.01, 1.0, 0.01):
        preds = (y_probs > t).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh_acc = t
            best_cm_acc = confusion_matrix(y_true, preds, labels=[0, 1])
    results['best_thresh_acc'] = best_thresh_acc
    results['max_acc'] = best_acc
    if best_cm_acc is not None:
        ConfusionMatrixDisplay(best_cm_acc, display_labels=["Real", "Fake"]).plot(cmap='Blues')
        plt.title(f"{display_title or ''} Threshold={best_thresh_acc:.2f} (Max Accuracy)")
        plt.show()
    return results


def load_metadata_from_json_paths(json_paths: List[str], fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Given a list of JSON file paths, load the specified fields from each JSON.
    If fields is None, load all fields.
    Returns a list of dicts (one per sample).
    """
    metadata = []
    for path in json_paths:
        if not os.path.exists(path):
            metadata.append({})
            continue
        with open(path, 'r') as f:
            data = json.load(f)
        if fields is not None:
            filtered = {k: data.get(k, None) for k in fields}
            metadata.append(filtered)
        else:
            metadata.append(data)
    return metadata 