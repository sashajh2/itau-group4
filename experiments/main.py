# experiments/main.py
"""
Main experiment runner for deepfake detection experiments.
Supports both single experiments and hyperparameter sweeps.
"""

import argparse
import json
from pathlib import Path
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
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        run_single_experiment(args.config, args.output_dir)
    elif args.mode == 'grid_search':
        run_hyperparameter_search(args.output_dir)

def run_single_experiment(config_path: str, output_dir: str):
    """Run a single experiment from config file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load data
    data_loaders = load_data()
    
    # Run experiment
    trainer = ModelTrainer()
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
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()