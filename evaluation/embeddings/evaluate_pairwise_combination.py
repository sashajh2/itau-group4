import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, f1_score, confusion_matrix
from itertools import combinations


def compute_pairwise_metrics(df, metrics_df, embedding_A, embedding_B):
    """
    Compute pairwise metrics for two embeddings.
    
    Args:
        df: DataFrame with probability columns and labels
        metrics_df: DataFrame with threshold information
        embedding_A: name of first embedding
        embedding_B: name of second embedding
        
    Returns:
        dict: dictionary containing all pairwise metrics
    """
    # Get thresholds
    threshold_A = metrics_df[metrics_df['embedding'] == embedding_A]['threshold'].iloc[0]
    threshold_B = metrics_df[metrics_df['embedding'] == embedding_B]['threshold'].iloc[0]
    
    # Get probabilities and labels
    prob_A = df[f'{embedding_A}_prob'].values
    prob_B = df[f'{embedding_B}_prob'].values
    y_true = df['label'].values
    
    # Predictions for individual embeddings
    pred_A = (prob_A >= threshold_A).astype(int)
    pred_B = (prob_B >= threshold_B).astype(int)
    
    # Hard OR prediction
    pred_A_or_B = ((prob_A >= threshold_A) | (prob_B >= threshold_B)).astype(int)
    
    # Individual metrics
    recall_A = recall_score(y_true, pred_A)
    recall_B = recall_score(y_true, pred_B)
    recall_A_or_B = recall_score(y_true, pred_A_or_B)
    
    fpr_A = 1 - specificity_score(y_true, pred_A)
    fpr_B = 1 - specificity_score(y_true, pred_B)
    fpr_A_or_B = 1 - specificity_score(y_true, pred_A_or_B)
    
    f1_A = f1_score(y_true, pred_A)
    f1_B = f1_score(y_true, pred_B)
    f1_A_or_B = f1_score(y_true, pred_A_or_B)
    
    # Slack metrics
    # Cases where A predicts below threshold but B predicts above threshold
    slack_B_picks_up_A = np.sum((pred_A == 0) & (pred_B == 1) & (y_true == 1))
    
    # Cases where B predicts below threshold but A predicts above threshold
    slack_A_picks_up_B = np.sum((pred_B == 0) & (pred_A == 1) & (y_true == 1))
    
    return {
        'embedding_A': embedding_A,
        'embedding_B': embedding_B,
        'recall_A': recall_A,
        'recall_A_or_B': recall_A_or_B,
        'recall_gain': recall_A_or_B - recall_A,
        'fpr_A': fpr_A,
        'fpr_A_or_B': fpr_A_or_B,
        'fp_increase': fpr_A_or_B - fpr_A,
        'f1_A': f1_A,
        'f1_A_or_B': f1_A_or_B,
        'f1_gain': f1_A_or_B - f1_A,
        'slack_B_picks_up_A': slack_B_picks_up_A,
        'slack_A_picks_up_B': slack_A_picks_up_B
    }


def specificity_score(y_true, y_pred):
    """Compute specificity (True Negative Rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate pairwise combinations of embeddings.")
    parser.add_argument('--probs_csv', type=str, required=True, 
                       help='Path to CSV with probability columns (e.g., audio_linear_probe_probs.csv)')
    parser.add_argument('--metrics_csv', type=str, required=True,
                       help='Path to CSV with embedding metrics including thresholds')
    parser.add_argument('--output', type=str, default='pairwise_audio_analysis.csv',
                       help='Output CSV file path')
    args = parser.parse_args()
    
    # Load DataFrames
    df = pd.read_csv(args.probs_csv)
    metrics_df = pd.read_csv(args.metrics_csv)
    
    # Get embedding names from probability columns
    prob_cols = [col for col in df.columns if col.endswith('_prob')]
    embeddings = [col.replace('_prob', '') for col in prob_cols]
    
    if len(embeddings) < 2:
        raise ValueError("Need at least 2 embeddings for pairwise analysis.")
    
    print(f"Found embeddings: {embeddings}")
    
    # Compute pairwise metrics for all combinations
    results = []
    
    for emb_A, emb_B in combinations(embeddings, 2):
        print(f"\nAnalyzing pair: {emb_A} vs {emb_B}")
        
        metrics = compute_pairwise_metrics(df, metrics_df, emb_A, emb_B)
        results.append(metrics)
        
        # Print summary
        print(f"  Recall gain: {metrics['recall_gain']:.4f}")
        print(f"  FPR increase: {metrics['fp_increase']:.4f}")
        print(f"  F1 gain: {metrics['f1_gain']:.4f}")
        print(f"  Slack B picks up A: {metrics['slack_B_picks_up_A']}")
        print(f"  Slack A picks up B: {metrics['slack_A_picks_up_B']}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved pairwise analysis to {args.output}")
    
    # Print overall summary
    print(f"\nOverall Summary:")
    print(f"Total pairs analyzed: {len(results)}")
    print(f"Average recall gain: {results_df['recall_gain'].mean():.4f}")
    print(f"Average FPR increase: {results_df['fp_increase'].mean():.4f}")
    print(f"Average F1 gain: {results_df['f1_gain'].mean():.4f}")


if __name__ == "__main__":
    main() 