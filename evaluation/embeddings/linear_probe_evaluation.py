import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve


def compute_metrics(y_true, y_probs):
    """
    Compute various metrics for binary classification.
    
    Args:
        y_true: true labels
        y_probs: predicted probabilities
        
    Returns:
        dict: dictionary containing all metrics
    """
    # ROC AUC
    auc = roc_auc_score(y_true, y_probs)
    
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    # Use threshold that maximizes TPR - FPR (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Predictions using optimal threshold
    y_pred = (y_probs >= optimal_threshold).astype(int)
    
    # Other metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # FPR (False Positive Rate)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'roc_auc': auc,
        'accuracy': accuracy,
        'recall': recall,
        'fpr': fpr_rate,
        'f1': f1,
        'threshold': optimal_threshold,
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate linear probe results and compute metrics.")
    parser.add_argument('--input', type=str, required=True, help='Path to CSV file (e.g., audio_linear_probe_probs.csv)')
    parser.add_argument('--output', type=str, default='audio_linear_probe_results.csv', help='Output CSV file path')
    args = parser.parse_args()

    # Load DataFrame
    df = pd.read_csv(args.input)
    
    # Get probability columns and labels
    prob_cols = [col for col in df.columns if col.endswith('_prob')]
    labels = df['label'].values
    
    if not prob_cols:
        raise ValueError("No columns ending with '_prob' found in the input CSV.")
    
    # Compute metrics for each embedding
    results = []
    
    for prob_col in prob_cols:
        embedding_name = prob_col.replace('_prob', '')
        y_probs = df[prob_col].values
        
        metrics = compute_metrics(labels, y_probs)
        
        # Add embedding name and format confusion matrix
        cm = metrics['confusion_matrix']
        cm_str = f"TN:{cm[0,0]}, FP:{cm[0,1]}, FN:{cm[1,0]}, TP:{cm[1,1]}"
        
        results.append({
            'embedding': embedding_name,
            'roc_auc': metrics['roc_auc'],
            'accuracy': metrics['accuracy'],
            'recall': metrics['recall'],
            'fpr': metrics['fpr'],
            'f1': metrics['f1'],
            'threshold': metrics['threshold'],
            'confusion_matrix': cm_str
        })
        
        print(f"\nResults for {embedding_name}:")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  FPR: {metrics['fpr']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Optimal Threshold: {metrics['threshold']:.4f}")
        print(f"  Confusion Matrix: {cm_str}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main() 