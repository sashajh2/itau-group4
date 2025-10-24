# experiments/grid_search.py
"""
Hyperparameter grid search for deepfake detection models.
"""

import json
from itertools import product
from pathlib import Path
from training.trainer import ModelTrainer
from data.data_loader import load_data

def run_hyperparameter_search(output_dir: str = './results'):
    """Run comprehensive hyperparameter search"""
    
    # Define search spaces
    model_configs = [
        {
            'type': 'orthogonal',
            'use_adapter': True,
            'adapter_hidden_dim': 256,
            'adapter_layers': 2,
            'head_dim': 128,
            'classifier_hidden_dim': 64
        },
        {
            'type': 'orthogonal', 
            'use_adapter': False,  # No adapter
            'adapter_hidden_dim': 512,
            'adapter_layers': 0,
            'head_dim': 128,
            'classifier_hidden_dim': 64
        },
        {
            'type': 'direct_classifier',
            'classifier_dims': [256, 128, 64]
        }
    ]
    
    training_configs = [
        {'stage_a': True, 'stage_b': True, 'stage_a_epochs': 3},
        {'stage_a': False, 'stage_b': True, 'stage_a_epochs': 0},
        {'stage_a': True, 'stage_b': True, 'stage_a_epochs': 10},
    ]
    
    evaluation_configs = [
        {'mahalanobis': True, 'linear_probe': True, 'mlp_classifier': True, 'knn': True},
        {'mahalanobis': True, 'linear_probe': False, 'mlp_classifier': True, 'knn': True},
    ]
    
    # Load data
    data_loaders = load_data()
    
    results = []
    
    for i, (model_config, training_config, eval_config) in enumerate(product(model_configs, training_configs, evaluation_configs)):
        print(f"Experiment {i+1}/{len(model_configs) * len(training_configs) * len(evaluation_configs)}")
        print(f"Model: {model_config['type']}, Training: {training_config}")
        
        trainer = ModelTrainer()
        result = trainer.train_and_evaluate(
            model_config, training_config, eval_config, data_loaders
        )
        
        # Store results
        experiment_result = {
            'experiment_id': i,
            'model_config': model_config,
            'training_config': training_config,
            'evaluation_config': eval_config,
            'results': result
        }
        results.append(experiment_result)
    
    # Save results
    output_path = Path(output_dir) / 'hyperparameter_search_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Grid search results saved to {output_path}")
    return results