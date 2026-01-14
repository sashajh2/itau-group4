"""
Detailed Model Testing Script

This script provides comprehensive testing with detailed statistics for multiple datasets.
It tests the trained model on AVDeepfake1M, ShareVeo3, and optionally Sora2 datasets separately,
providing per-dataset metrics, confusion matrices, and detailed analysis.
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

# Import from time_series_model
import sys
import os
# Add common paths where time_series_model.py might be located
possible_paths = [
    os.path.dirname(os.path.abspath(__file__)),  # Same directory as this script
    '/content/models',  # Colab workspace
    '.',
]
for path in possible_paths:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from time_series_model import (
        AVH5Dataset,
        AVTemporalModel,
        ModelConfig,
        collate_fn,
    )
except ImportError:
    # If import fails, try to find the file
    import glob
    for path in possible_paths:
        model_file = os.path.join(path, 'time_series_model.py')
        if os.path.exists(model_file):
            sys.path.insert(0, path)
            from time_series_model import (
                AVH5Dataset,
                AVTemporalModel,
                ModelConfig,
                collate_fn,
            )
            break
    else:
        raise ImportError("Could not find time_series_model.py. Make sure it's in the same directory or in /content/models/")


def test_model_detailed(
    model: AVTemporalModel,
    test_loader: DataLoader,
    device: torch.device,
    dataset_name: str = "test",
    save_predictions: bool = False,
) -> Dict:
    """
    Test model with detailed statistics.
    
    Args:
        model: The model to test
        test_loader: DataLoader for test data
        device: Device to run on
        dataset_name: Name of the dataset (for reporting)
        save_predictions: Whether to save per-sample predictions
    
    Returns:
        Dictionary with detailed test metrics
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    total_loss = 0.0
    all_logits = []
    all_labels = []
    all_probs = []
    all_preds = []
    
    # Per-batch statistics
    batch_stats = []
    
    # Per-sample tracking (if save_predictions)
    sample_predictions = [] if save_predictions else None
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Testing on {dataset_name}")
        batch_idx = 0
        
        for batch in pbar:
            audio_seq = batch["audio_seq"].to(device)
            video_seq = batch["video_seq"].to(device)
            label_seq = batch["label_seq"].to(device)
            valid_length = batch.get("valid_length", None)
            if valid_length is not None:
                valid_length = valid_length.to(device)
            
            video_ids = batch.get("video_id", [])
            aug_indices = batch.get("aug_idx", [])
            
            # Forward pass
            segment_logits, _ = model(audio_seq, video_seq, valid_length=valid_length)
            
            # Create mask for valid segments
            if valid_length is not None:
                mask = torch.zeros_like(label_seq, dtype=torch.bool)
                for b in range(label_seq.shape[0]):
                    mask[b, :valid_length[b]] = True
            else:
                mask = torch.ones_like(label_seq, dtype=torch.bool)
            
            # Mask out padding in loss computation
            segment_logits_masked = segment_logits.masked_fill(~mask, 0.0)
            label_seq_masked = label_seq.masked_fill(~mask, 0.0)
            
            # Compute loss
            loss = criterion(segment_logits_masked, label_seq_masked)
            total_loss += loss.item()
            
            # Store logits and labels (only valid segments)
            batch_logits = []
            batch_labels = []
            
            for b in range(label_seq.shape[0]):
                if valid_length is not None:
                    valid_len = valid_length[b].item()
                    batch_logits.append(segment_logits[b, :valid_len].cpu().numpy())
                    batch_labels.append(label_seq[b, :valid_len].cpu().numpy())
                else:
                    batch_logits.append(segment_logits[b].cpu().numpy())
                    batch_labels.append(label_seq[b].cpu().numpy())
                
                # Store per-sample predictions if requested
                if save_predictions:
                    probs_b = torch.sigmoid(segment_logits[b, :valid_len] if valid_length else segment_logits[b]).cpu().numpy()
                    preds_b = (probs_b > 0.5).astype(int)
                    labels_b = batch_labels[-1]
                    
                    sample_predictions.append({
                        'video_id': video_ids[b] if b < len(video_ids) else f'unknown_{b}',
                        'aug_idx': aug_indices[b] if b < len(aug_indices) else -1,
                        'num_segments': len(labels_b),
                        'mean_prob': float(np.mean(probs_b)),
                        'mean_pred': float(np.mean(preds_b)),
                        'mean_label': float(np.mean(labels_b)),
                        'accuracy': float(np.mean(preds_b == labels_b)),
                    })
            
            # Flatten for batch-level metrics
            batch_logits_flat = np.concatenate([arr.flatten() for arr in batch_logits])
            batch_labels_flat = np.concatenate([arr.flatten() for arr in batch_labels])
            batch_probs = 1 / (1 + np.exp(-batch_logits_flat))  # Sigmoid
            batch_preds = (batch_probs > 0.5).astype(int)
            batch_labels_binary = (batch_labels_flat > 0.5).astype(int)
            
            # Batch-level metrics
            batch_accuracy = accuracy_score(batch_labels_binary, batch_preds)
            batch_precision = precision_score(batch_labels_binary, batch_preds, zero_division=0)
            batch_recall = recall_score(batch_labels_binary, batch_preds, zero_division=0)
            batch_f1 = f1_score(batch_labels_binary, batch_preds, zero_division=0)
            
            try:
                batch_auroc = roc_auc_score(batch_labels_binary, batch_logits_flat) if len(np.unique(batch_labels_binary)) > 1 else 0.0
            except:
                batch_auroc = 0.0
            
            batch_stats.append({
                'batch_idx': batch_idx,
                'loss': float(loss.item()),
                'num_samples': len(batch_logits),
                'num_segments': len(batch_logits_flat),
                'accuracy': float(batch_accuracy),
                'precision': float(batch_precision),
                'recall': float(batch_recall),
                'f1': float(batch_f1),
                'auroc': float(batch_auroc),
            })
            
            # Accumulate for overall metrics
            all_logits.extend(batch_logits)
            all_labels.extend(batch_labels)
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': batch_accuracy,
                'auroc': batch_auroc if not np.isnan(batch_auroc) else 0.0
            })
            
            batch_idx += 1
    
    # Compute overall metrics
    all_logits_flat = np.concatenate([arr.flatten() for arr in all_logits])
    all_labels_flat = np.concatenate([arr.flatten() for arr in all_labels])
    
    # Convert to binary
    all_labels_binary = (all_labels_flat > 0.5).astype(int)
    all_probs = 1 / (1 + np.exp(-all_logits_flat))  # Sigmoid
    all_preds = (all_probs > 0.5).astype(int)
    
    # Overall metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels_binary, all_preds)
    precision = precision_score(all_labels_binary, all_preds, zero_division=0)
    recall = recall_score(all_labels_binary, all_preds, zero_division=0)
    f1 = f1_score(all_labels_binary, all_preds, zero_division=0)
    
    try:
        auroc = roc_auc_score(all_labels_binary, all_logits_flat) if len(np.unique(all_labels_binary)) > 1 else 0.0
    except ValueError:
        auroc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels_binary, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, cm[0, 0] if cm.size == 1 else 0)
    
    # Classification report
    class_report = classification_report(
        all_labels_binary, all_preds,
        target_names=['Fake', 'Real'],
        output_dict=True,
        zero_division=0
    )
    
    # Label distribution
    unique_labels, label_counts = np.unique(all_labels_binary, return_counts=True)
    label_dist = {int(label): int(count) for label, count in zip(unique_labels, label_counts)}
    
    # Prediction distribution
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    pred_dist = {int(pred): int(count) for pred, count in zip(unique_preds, pred_counts)}
    
    # Statistics about probabilities
    prob_stats = {
        'mean': float(np.mean(all_probs)),
        'std': float(np.std(all_probs)),
        'min': float(np.min(all_probs)),
        'max': float(np.max(all_probs)),
        'median': float(np.median(all_probs)),
        'q25': float(np.percentile(all_probs, 25)),
        'q75': float(np.percentile(all_probs, 75)),
    }
    
    # Per-class probability statistics
    prob_stats_by_class = {}
    for label_val in unique_labels:
        mask = all_labels_binary == label_val
        if np.sum(mask) > 0:
            prob_stats_by_class[f'class_{label_val}'] = {
                'mean': float(np.mean(all_probs[mask])),
                'std': float(np.std(all_probs[mask])),
                'median': float(np.median(all_probs[mask])),
            }
    
    results = {
        'dataset_name': dataset_name,
        'num_samples': len(all_logits),
        'num_segments': len(all_logits_flat),
        'metrics': {
            'loss': float(avg_loss),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auroc': float(auroc),
        },
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
        },
        'classification_report': class_report,
        'label_distribution': label_dist,
        'prediction_distribution': pred_dist,
        'probability_statistics': prob_stats,
        'probability_by_class': prob_stats_by_class,
        'batch_statistics': batch_stats,
    }
    
    if save_predictions and sample_predictions:
        results['sample_predictions'] = sample_predictions[:1000]  # Limit to first 1000 for file size
    
    return results


def print_results(results: Dict, verbose: bool = True):
    """Print formatted results."""
    print("\n" + "="*70)
    print(f"TEST RESULTS: {results['dataset_name'].upper()}")
    print("="*70)
    
    print(f"\nDataset Info:")
    print(f"  Number of samples: {results['num_samples']:,}")
    print(f"  Number of segments: {results['num_segments']:,}")
    
    print(f"\nOverall Metrics:")
    metrics = results['metrics']
    print(f"  Loss:           {metrics['loss']:.4f}")
    print(f"  Accuracy:       {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  F1 Score:       {metrics['f1_score']:.4f}")
    print(f"  AUROC:          {metrics['auroc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = results['confusion_matrix']
    print(f"                  Predicted")
    print(f"                Fake    Real")
    print(f"  Actual Fake   {cm['true_negative']:5d}   {cm['false_positive']:5d}")
    print(f"         Real   {cm['false_negative']:5d}   {cm['true_positive']:5d}")
    
    print(f"\nLabel Distribution:")
    for label, count in results['label_distribution'].items():
        label_name = "Real" if label == 1 else "Fake"
        pct = 100 * count / results['num_segments']
        print(f"  {label_name}: {count:,} ({pct:.2f}%)")
    
    print(f"\nPrediction Distribution:")
    for pred, count in results['prediction_distribution'].items():
        pred_name = "Real" if pred == 1 else "Fake"
        pct = 100 * count / results['num_segments']
        print(f"  {pred_name}: {count:,} ({pct:.2f}%)")
    
    print(f"\nProbability Statistics:")
    prob_stats = results['probability_statistics']
    print(f"  Mean:    {prob_stats['mean']:.4f}")
    print(f"  Std:     {prob_stats['std']:.4f}")
    print(f"  Min:     {prob_stats['min']:.4f}")
    print(f"  Max:     {prob_stats['max']:.4f}")
    print(f"  Median:  {prob_stats['median']:.4f}")
    print(f"  Q25:     {prob_stats['q25']:.4f}")
    print(f"  Q75:     {prob_stats['q75']:.4f}")
    
    if verbose and len(results.get('batch_statistics', [])) > 0:
        print(f"\nBatch Statistics (first 5 batches):")
        for batch_stat in results['batch_statistics'][:5]:
            print(f"  Batch {batch_stat['batch_idx']}: "
                  f"Loss={batch_stat['loss']:.4f}, "
                  f"Acc={batch_stat['accuracy']:.4f}, "
                  f"AUROC={batch_stat['auroc']:.4f}, "
                  f"Segments={batch_stat['num_segments']}")
    
    if verbose and len(results.get('batch_statistics', [])) > 0:
        # Summary of batch statistics
        batch_accs = [b['accuracy'] for b in results['batch_statistics']]
        batch_aurocs = [b['auroc'] for b in results['batch_statistics']]
        print(f"\nBatch Statistics Summary:")
        print(f"  Accuracy:  mean={np.mean(batch_accs):.4f}, std={np.std(batch_accs):.4f}")
        print(f"  AUROC:     mean={np.mean(batch_aurocs):.4f}, std={np.std(batch_aurocs):.4f}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Detailed model testing on multiple datasets")
    parser.add_argument("--hdf5_path", type=str, required=True,
                       help="Path to HDF5 file")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint (.pt file)")
    parser.add_argument("--datasets", type=str, nargs='+', 
                       default=['avdeepfake1m', 'shareveo3'],
                       choices=['avdeepfake1m', 'shareveo3', 'sora2', 'all'],
                       help="Datasets to test on")
    parser.add_argument("--audio_embedding", type=str, default="openl3",
                       choices=['openl3', 'hubert'],
                       help="Audio embedding type")
    parser.add_argument("--video_embedding", type=str, default="senet",
                       help="Video embedding type")
    parser.add_argument("--use_audio_labels", action='store_true', default=True,
                       help="Use audio labels (default: True)")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for testing")
    parser.add_argument("--output_dir", type=str, default="./test_results",
                       help="Directory to save results")
    parser.add_argument("--save_predictions", action='store_true',
                       help="Save per-sample predictions (first 1000 samples)")
    parser.add_argument("--save_batch_stats", action='store_true', default=True,
                       help="Save per-batch statistics")
    
    args = parser.parse_args()
    
    # Determine which datasets to test
    if 'all' in args.datasets:
        datasets_to_test = ['avdeepfake1m', 'shareveo3', 'sora2']
    else:
        datasets_to_test = args.datasets
    
    # Configuration
    config = ModelConfig(
        audio_emb_dim=512,
        video_emb_dim=2048,
        patch_size=8,
        patch_stride=4,
        model_dim=256,
        num_heads=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        learning_rate=1e-4,
        weight_decay=1e-5,
        batch_size=args.batch_size,
        num_epochs=1,
        audio_embedding_type=args.audio_embedding,
        video_embedding_type=args.video_embedding,
        use_audio_labels=args.use_audio_labels,
    )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    
    # Create model and load weights
    model = AVTemporalModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    if 'val_auroc' in checkpoint:
        print(f"  Checkpoint validation AUROC: {checkpoint['val_auroc']:.4f}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test on each dataset
    all_results = {}
    
    for dataset_name in datasets_to_test:
        print("\n" + "="*70)
        print(f"Testing on {dataset_name.upper()}")
        print("="*70)
        
        # Load dataset
        test_dataset = AVH5Dataset(
            hdf5_path=args.hdf5_path,
            audio_embedding_type=config.audio_embedding_type,
            video_embedding_type=config.video_embedding_type,
            use_audio_labels=config.use_audio_labels,
            video_ids=None,
            filter_dataset=dataset_name,
        )
        
        if len(test_dataset) == 0:
            print(f"⚠️  No samples found for dataset '{dataset_name}'. Skipping.")
            continue
        
        print(f"Loaded {len(test_dataset)} samples")
        
        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if device.type == "cuda" else False,
        )
        
        # Test model
        results = test_model_detailed(
            model=model,
            test_loader=test_loader,
            device=device,
            dataset_name=dataset_name,
            save_predictions=args.save_predictions,
        )
        
        # Print results
        print_results(results, verbose=True)
        
        # Save results to JSON
        if not args.save_batch_stats:
            results.pop('batch_statistics', None)
        
        output_file = os.path.join(args.output_dir, f"test_results_{dataset_name}_{timestamp}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
        
        all_results[dataset_name] = results
    
    # Print summary across all datasets
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("SUMMARY ACROSS ALL DATASETS")
        print("="*70)
        print(f"\n{'Dataset':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUROC':<12}")
        print("-"*70)
        for dataset_name, results in all_results.items():
            m = results['metrics']
            print(f"{dataset_name:<20} {m['accuracy']:>10.4f}  {m['precision']:>10.4f}  "
                  f"{m['recall']:>10.4f}  {m['f1_score']:>10.4f}  {m['auroc']:>10.4f}")
        
        # Overall average (weighted by number of segments)
        total_segments = sum(r['num_segments'] for r in all_results.values())
        weighted_metrics = {}
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']:
            weighted_metrics[metric_name] = sum(
                r['metrics'][metric_name] * r['num_segments'] 
                for r in all_results.values()
            ) / total_segments
        
        print("-"*70)
        print(f"{'Weighted Average':<20} {weighted_metrics['accuracy']:>10.4f}  "
              f"{weighted_metrics['precision']:>10.4f}  {weighted_metrics['recall']:>10.4f}  "
              f"{weighted_metrics['f1_score']:>10.4f}  {weighted_metrics['auroc']:>10.4f}")
        print("="*70)
        
        # Save combined results
        combined_output = os.path.join(args.output_dir, f"test_results_combined_{timestamp}.json")
        combined_results = {
            'timestamp': timestamp,
            'checkpoint_path': args.checkpoint_path,
            'datasets_tested': list(all_results.keys()),
            'per_dataset_results': all_results,
            'weighted_averages': weighted_metrics,
        }
        with open(combined_output, 'w') as f:
            json.dump(combined_results, f, indent=2)
        print(f"\n✓ Combined results saved to: {combined_output}")


if __name__ == "__main__":
    main()

