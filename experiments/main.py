# experiments/main.py
"""
Main experiment runner for deepfake detection experiments.
Supports both single experiments and hyperparameter sweeps.
"""

import argparse
import json
from pathlib import Path
import torch

from training.trainer import ModelTrainer
from data.data_loader import load_data
from experiments.grid_search import run_hyperparameter_search


def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Experiments')
    parser.add_argument('--mode', choices=['single', 'grid_search'], default='single',
                       help='Run single experiment or hyperparameter search')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment configuration file')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker processes for DataLoader')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        run_single_experiment(args.config, args.output_dir, args.device, args.batch_size, args.num_workers)
    elif args.mode == 'grid_search':
        run_hyperparameter_search(args.output_dir)


def run_single_experiment(config_path: str, output_dir: str, device: str = 'cuda',
                          batch_size: int = 256, num_workers: int = 4):
    """Run a single experiment from config file"""
    print(f"Loading experiment configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract data config
    data_config = config.get('data_config', {})
    model_name = data_config.get('model_name', 'openl3')
    version = data_config.get('version', '2025-09-12')
    noise = data_config.get('noise', 'none')
    denoiser_name = data_config.get('denoiser_name', 'none')
    
    print(f"\n{'='*60}")
    print(f"Experiment: {config.get('name', 'unnamed')}")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Version: {version}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Load data from Neon
    print("Loading embeddings from Neon Postgres...")
    data_loaders = load_data(
        model_name=model_name,
        version=version,
        noise=noise,
        denoiser_name=denoiser_name,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Run experiment
    print("\nStarting training...")
    trainer = ModelTrainer(device=device)
    results = trainer.train_and_evaluate(
        config['model_config'],
        config['training_config'], 
        config['evaluation_config'],
        data_loaders
    )
    
    # Save results
    output_path = Path(output_dir) / f"experiment_{config.get('name', 'unnamed')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'config': config,
            'results': results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Results saved to {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
