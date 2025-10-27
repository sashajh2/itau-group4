"""
Convenience script to run experiments from command line.

Example:

# Run orthogonal model experiment
python experiments/main.py --mode single --config models/configs/orthogonal_model.json

# Or use convenience script
python experiments/run_experiment.py orthogonal_model

# Run hyperparameter search
python experiments/run_experiment.py grid_search
 
or 

python experiments/main.py --mode grid_search --output_dir ./results

# Testing direct classifier
python experiments/run_experiment.py direct_classifier
"""

import subprocess
import sys
from pathlib import Path

def run_experiment(config_name: str, mode: str = 'single'):
    """Run an experiment with a predefined config"""
    config_path = Path('models/configs') / f"{config_name}.json"
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    
    cmd = [
        sys.executable, 'experiments/main.py',
        '--mode', mode,
        '--config', str(config_path),
        '--output_dir', './results'
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config name (without .json)')
    parser.add_argument('--mode', choices=['single', 'grid_search'], default='single')
    
    args = parser.parse_args()
    run_experiment(args.config, args.mode)