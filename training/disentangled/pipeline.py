#!/usr/bin/env python3
"""
Unified training and evaluation pipeline for disentangled representation learning.

This script:
1. Loads training and test datasets (once)
2. Loops over hyperparameter configurations (Conservative/Moderate/Aggressive)
3. For each config:
   - Trains model
   - Evaluates on training dataset (all metrics including local content-group)
   - Evaluates on test dataset (cross-dataset metrics, no local content-group)
4. Saves all results
"""
import argparse
import os
import json
import numpy as np
import torch
from typing import Dict

from training.disentangled.data_utils import load_data_from_hdf5
from training.disentangled.train_utils import train_model
from training.disentangled.eval_utils import evaluate_single_dataset, evaluate_cross_dataset


# Hyperparameter configurations
HYPERPARAMETER_CONFIGS = {
    'conservative': {
        'min_variance': 0.1,
        'variance_reg_weight': 1.0,
        'lambda_var': 0.5,
        'lambda_orth': 0.1,
        'temperature': 0.1,
        'min_orth': 0.001,
        'use_equal_weight_normalization': True,
        'use_adaptive_scaling': False,
        'min_loss_ratio': 0.1,
    },
    'lambda_var_0_0': {
        'min_variance': 0.1,
        'variance_reg_weight': 1.0,
        'lambda_var': 0.0,
        'lambda_orth': 0.1,
        'temperature': 0.1,
        'min_orth': 0.001,
        'use_equal_weight_normalization': False,
        'use_adaptive_scaling': False,
        'min_loss_ratio': 0.1,
    },
    'lambda_var_0_6': {
        'min_variance': 0.1,
        'variance_reg_weight': 1.0,
        'lambda_var': 0.6,
        'lambda_orth': 0.1,
        'temperature': 0.1,
        'min_orth': 0.001,
        'use_equal_weight_normalization': False,
        'use_adaptive_scaling': False,
        'min_loss_ratio': 0.1,
    },
    'lambda_var_0_1': {
        'min_variance': 0.1,
        'variance_reg_weight': 1.0,
        'lambda_var': 0.1,
        'lambda_orth': 0.1,
        'temperature': 0.1,
        'min_orth': 0.001,
        'use_equal_weight_normalization': False,
        'use_adaptive_scaling': False,
        'min_loss_ratio': 0.1,
    },
    'lambda_var_0_2': {
        'min_variance': 0.1,
        'variance_reg_weight': 1.0,
        'lambda_var': 0.2,
        'lambda_orth': 0.1,
        'temperature': 0.1,
        'min_orth': 0.001,
        'use_equal_weight_normalization': False,
        'use_adaptive_scaling': False,
        'min_loss_ratio': 0.1,
    },
    'lambda_var_0_3': {
        'min_variance': 0.1,
        'variance_reg_weight': 1.0,
        'lambda_var': 0.3,
        'lambda_orth': 0.1,
        'temperature': 0.1,
        'min_orth': 0.001,
        'use_equal_weight_normalization': False,
        'use_adaptive_scaling': False,
        'min_loss_ratio': 0.1,
    },
    'lambda_var_0_4': {
        'min_variance': 0.1,
        'variance_reg_weight': 1.0,
        'lambda_var': 0.4,
        'lambda_orth': 0.1,
        'temperature': 0.1,
        'min_orth': 0.001,
        'use_equal_weight_normalization': False,
        'use_adaptive_scaling': False,
        'min_loss_ratio': 0.1,
    },
    'moderate': {
        'min_variance': 0.2,
        'variance_reg_weight': 2.0,
        'lambda_var': 0.5,
        'lambda_orth': 0.1,
        'temperature': 0.1,
        'min_orth': 0.001,
        'use_equal_weight_normalization': True,
        'use_adaptive_scaling': False,
        'min_loss_ratio': 0.1,
    },
    'aggressive': {
        'min_variance': 0.5,
        'variance_reg_weight': 5.0,
        'lambda_var': 0.5,
        'lambda_orth': 0.1,
        'temperature': 0.1,
        'min_orth': 0.001,
        'use_equal_weight_normalization': True,
        'use_adaptive_scaling': False,
        'min_loss_ratio': 0.1,
    }
}


def print_summary_table(all_results: Dict):
    """Print summary comparison table of all configs."""
    print("\n" + "="*100)
    print("HYPERPARAMETER SWEEP SUMMARY")
    print("="*100)
    
    # Key metrics to compare
    key_metrics = [
        ('train_model_metrics', 'wasserstein_distance', 'Wasserstein Distance (Train)'),
        ('train_model_metrics', 'separation_gap', 'Separation Gap (Train)'),
        ('cross_dataset_metrics', 'wasserstein_distance_cross_dataset_z_auth', 'Wasserstein Distance (Cross-Dataset)'),
        ('cross_dataset_metrics', 'separation_gap_cross_dataset_z_auth', 'Separation Gap (Cross-Dataset)'),
    ]
    
    print(f"\n{'Metric':<50} {'Conservative':>15} {'Moderate':>15} {'Aggressive':>15}")
    print("-" * 100)
    
    for result_key, metric_key, label in key_metrics:
        values = []
        for config_name in ['conservative', 'moderate', 'aggressive']:
            if config_name in all_results:
                config_results = all_results[config_name]
                if result_key in config_results and metric_key in config_results[result_key]:
                    values.append(config_results[result_key][metric_key])
                else:
                    values.append(None)
            else:
                values.append(None)
        
        if any(v is not None for v in values):
            print(f"{label:<50}", end="")
            for v in values:
                if v is not None:
                    print(f"{v:>15.4f}", end="")
                else:
                    print(f"{'N/A':>15}", end="")
            print()
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(
        description="Unified training and evaluation pipeline for disentangled representation learning"
    )
    
    # Data arguments
    parser.add_argument('--train-hdf5', type=str,
                       default='exports/deepfake_embeddings_2.h5',
                       help='Path to training dataset HDF5 file')
    parser.add_argument('--test-hdf5', type=str,
                       default='exports/sora2_embeddings.h5',
                       help='Path to test dataset HDF5 file (for cross-dataset evaluation)')
    parser.add_argument('--encoder-name', type=str, default='hubert',
                       choices=['hubert', 'openl3', 'senet'],
                       help='Encoder to use (default: hubert)')
    
    # Model arguments
    parser.add_argument('--input-dim', type=int, default=768,
                       help='Input embedding dimension (default: 768 for hubert)')
    parser.add_argument('--output-dim', type=int, default=128,
                       help='Output projection dimension (default: 128)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio (default: 0.1)')
    
    # Evaluation arguments
    parser.add_argument('--eval-batch-size', type=int, default=256,
                       help='Batch size for evaluation inference (default: 256)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (None = all)')
    
    # Pipeline arguments
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save all results and checkpoints')
    parser.add_argument('--run-hyperparameter-sweep', action='store_true',
                       help='Run all 3 hyperparameter configs (Conservative/Moderate/Aggressive)')
    parser.add_argument('--config', type=str, default='conservative',
                       choices=[
                           'conservative', 'moderate', 'aggressive',
                           'lambda_var_0_0', 'lambda_var_0_1', 'lambda_var_0_2', 'lambda_var_0_3', 'lambda_var_0_4', 'lambda_var_0_6'
                       ],
                       help='Single config to run (if not running sweep)')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*100)
    print("UNIFIED TRAINING AND EVALUATION PIPELINE")
    print("="*100)
    print(f"Training HDF5: {args.train_hdf5}")
    print(f"Test HDF5: {args.test_hdf5}")
    print(f"Encoder: {args.encoder_name}")
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    if args.run_hyperparameter_sweep:
        print(f"Mode: Hyperparameter sweep (all 3 configs)")
    else:
        print(f"Mode: Single config ({args.config})")
    print("="*100)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========================================================================
    # STEP 1: LOAD DATASETS (ONCE)
    # ========================================================================
    print("\n" + "="*100)
    print("STEP 1: LOADING DATASETS")
    print("="*100)
    
    train_embeddings, train_labels, train_content_groups = load_data_from_hdf5(
        args.train_hdf5, args.encoder_name, args.max_samples
    )
    
    test_embeddings, test_labels, test_content_groups = load_data_from_hdf5(
        args.test_hdf5, args.encoder_name, args.max_samples
    )
    
    # Separate training real for cross-dataset evaluation
    train_real_mask = train_labels == 0
    train_real_embeddings = train_embeddings[train_real_mask]
    
    print(f"\n✅ Datasets loaded:")
    print(f"   Training: {len(train_embeddings):,} samples ({np.sum(train_labels == 0):,} real, {np.sum(train_labels == 1):,} fake)")
    print(f"   Training real (for cross-dataset): {len(train_real_embeddings):,} samples")
    print(f"   Test: {len(test_embeddings):,} samples")
    
    # ========================================================================
    # STEP 2: DETERMINE CONFIGS TO RUN
    # ========================================================================
    if args.run_hyperparameter_sweep:
        configs_to_run = HYPERPARAMETER_CONFIGS
    else:
        configs_to_run = {args.config: HYPERPARAMETER_CONFIGS[args.config]}
    
    # ========================================================================
    # STEP 3: LOOP OVER CONFIGS
    # ========================================================================
    all_results = {}
    
    for config_name, config in configs_to_run.items():
        print("\n" + "="*100)
        print(f"CONFIG: {config_name.upper()}")
        print("="*100)
        print(f"  min_variance: {config['min_variance']}")
        print(f"  variance_reg_weight: {config['variance_reg_weight']}")
        print(f"  lambda_var: {config['lambda_var']}")
        print(f"  lambda_orth: {config['lambda_orth']}")
        
        # Create save directory for this config
        config_save_dir = os.path.join(args.output_dir, config_name)
        os.makedirs(config_save_dir, exist_ok=True)
        
        # 3a. Train model
        print(f"\n{'='*100}")
        print(f"TRAINING MODEL ({config_name})")
        print(f"{'='*100}")
        
        checkpoint_path = train_model(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            train_content_groups=train_content_groups,
            config=config,
            save_dir=config_save_dir,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            num_workers=args.num_workers,
            val_split=args.val_split,
            device=device,
        )
        
        print(f"\n✅ Model trained and saved to: {checkpoint_path}")
        
        # 3b. Evaluate on training set
        print(f"\n{'='*100}")
        print(f"EVALUATING ON TRAINING DATASET ({config_name})")
        print(f"{'='*100}")
        
        train_results = evaluate_single_dataset(
            embeddings=train_embeddings,
            labels=train_labels,
            content_groups=train_content_groups,
            checkpoint_path=checkpoint_path,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            batch_size=args.eval_batch_size,
            device=device,
            verbose=True
        )
        
        # 3c. Evaluate on test set (cross-dataset)
        print(f"\n{'='*100}")
        print(f"EVALUATING CROSS-DATASET GENERALIZATION ({config_name})")
        print(f"{'='*100}")
        
        cross_dataset_results = evaluate_cross_dataset(
            train_real_embeddings=train_real_embeddings,
            test_embeddings=test_embeddings,
            checkpoint_path=checkpoint_path,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            batch_size=args.eval_batch_size,
            device=device,
            verbose=True
        )
        
        # Store results
        all_results[config_name] = {
            'config': config,
            'checkpoint_path': checkpoint_path,
            'train_metrics': train_results,
            'cross_dataset_metrics': cross_dataset_results,
        }
        
        # Save individual config results
        config_results_path = os.path.join(config_save_dir, 'results.json')
        with open(config_results_path, 'w') as f:
            json.dump(all_results[config_name], f, indent=2)
        
        print(f"\n💾 Results saved to: {config_results_path}")
    
    # ========================================================================
    # STEP 4: SAVE COMBINED RESULTS
    # ========================================================================
    combined_results_path = os.path.join(args.output_dir, 'all_results.json')
    with open(combined_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Combined results saved to: {combined_results_path}")
    
    # ========================================================================
    # STEP 5: PRINT SUMMARY
    # ========================================================================
    if args.run_hyperparameter_sweep and len(all_results) > 1:
        print_summary_table(all_results)
    
    print("\n✅ Pipeline complete!")


if __name__ == '__main__':
    main()
