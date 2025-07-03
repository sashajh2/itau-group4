import numpy as np  # type: ignore
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from collections import defaultdict
import json
from typing import List, Dict, Any, Optional
import os
import argparse
import pickle

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

def k_fold_linear_probe(
    embeddings, labels, metadata=None, stratify_by=None, folds=5, random_state=42
):
    """
    Run k-fold cross-validated linear probe (logistic regression) and evaluate.
    Returns a list of results (one per fold).
    """
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.model_selection import StratifiedKFold  # type: ignore
    from embeddings.embedding_eval import evaluate_model

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_train, y_train)
        y_probs = clf.predict_proba(X_test)[:, 1]
        meta_test = [metadata[i] for i in test_idx] if metadata is not None else None
        eval_results = evaluate_model(y_test, y_probs, metadata=meta_test, display_title=f"Fold {fold_idx+1}", stratify_by=stratify_by)
        results.append(eval_results)
        print(f"Fold {fold_idx+1} results:", eval_results['global'])
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate embeddings with stratified metrics.")
    parser.add_argument('--embeddings', type=str, required=True, help='Path to embeddings.npy')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels.pkl')
    parser.add_argument('--metadata', type=str, default=None, help='Path to metadata.pkl')
    parser.add_argument('--stratify_by', type=str, nargs='*', default=None, help='Metadata fields to stratify by')
    parser.add_argument('--output', type=str, default=None, help='Optional: Path to save results as .pkl')
    parser.add_argument('--probe_type', type=str, default='linear', help='Type of probe to use (default: linear)')
    parser.add_argument('--folds', type=int, default=5, help='Number of cross-validation folds')
    args = parser.parse_args()

    print(f"Loading embeddings from .npy file: {args.embeddings}")
    embeddings = np.load(args.embeddings)

    with open(args.labels, 'rb') as f:
        labels = pickle.load(f)

    metadata = None
    if args.metadata:
        with open(args.metadata, 'rb') as f:
            metadata = pickle.load(f)

    if args.probe_type == 'linear':
        results = k_fold_linear_probe(
            embeddings, labels, metadata=metadata, stratify_by=args.stratify_by, folds=args.folds
        )
    else:
        raise NotImplementedError(f"Probe type '{args.probe_type}' is not implemented.")

    if args.output:
        with open(args.output, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
