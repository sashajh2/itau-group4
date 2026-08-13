# Pipeline Refactoring Plan

## Goals

1. **Unified pipeline** that trains and evaluates in one run
2. **Avoid code duplication** - shared data loading and evaluation functions
3. **Support hyperparameter sweeps** - test Conservative/Moderate/Aggressive configs
4. **Focus on variance regularization** - no separation loss (anomaly detection approach)
5. **Modular design** - reusable components

---

## Architecture Overview

```
training/disentangled/
├── data_utils.py          (NEW) - Shared data loading functions
├── train_utils.py         (NEW) - Reusable training function
├── eval_utils.py          (NEW) - Reusable evaluation functions
├── pipeline.py            (NEW) - Unified training + evaluation pipeline
├── main.py                (EXISTING) - Can call train_utils
├── evaluate_metrics.py    (EXISTING) - Can call eval_utils
└── evaluate_cross_dataset.py (EXISTING) - Can call eval_utils
```

---

## Code Flow

```
pipeline.py (main entry point)
│
├── 1. Load datasets (once, shared)
│   └── data_utils.load_data_from_hdf5() for train and test
│
├── 2. Loop over hyperparameter configs (Conservative/Moderate/Aggressive)
│   │
│   ├── 2a. Train model
│   │   └── train_utils.train_model() 
│   │       - Takes pre-loaded dataset
│   │       - Returns checkpoint path
│   │
│   ├── 2b. Evaluate on training set
│   │   └── eval_utils.evaluate_single_dataset()
│   │       - Takes pre-loaded embeddings, labels, content_groups
│   │       - Takes model checkpoint path
│   │       - Returns metrics dict
│   │
│   └── 2c. Evaluate on test set (cross-dataset)
│       └── eval_utils.evaluate_cross_dataset()
│           - Takes pre-loaded train and test data
│           - Takes model checkpoint path
│           - Returns cross-dataset metrics dict
│
└── 3. Save all results
    └── JSON with all configs and their results
```

---

## File Structure

### 1. `data_utils.py` (NEW)

**Purpose**: Shared data loading functions

**Functions**:
```python
def load_data_from_hdf5(
    hdf5_path: str,
    encoder_name: str = 'hubert',
    max_samples: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings, labels, and content groups from HDF5.
    
    Returns:
        embeddings: [n_samples, emb_dim]
        labels: [n_samples] (0=real, 1=fake)
        content_groups: [n_samples] (content group IDs)
    """
    # Extract from evaluate_metrics.py load_data_from_hdf5()
```

**Usage**: Called once at pipeline start, results passed to training/evaluation

---

### 2. `train_utils.py` (NEW)

**Purpose**: Reusable training function

**Functions**:
```python
def train_model(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    train_content_groups: np.ndarray,
    config: Dict,  # Hyperparameters
    save_dir: str,
    device: str = 'cuda'
) -> str:
    """
    Train disentanglement model.
    
    Args:
        train_embeddings: Pre-loaded embeddings
        train_labels: Pre-loaded labels
        train_content_groups: Pre-loaded content groups
        config: Dict with hyperparameters (min_variance, variance_reg_weight, etc.)
        save_dir: Directory to save checkpoint
        device: Device to train on
    
    Returns:
        checkpoint_path: Path to saved model
    """
    # Extract training logic from main.py
    # Create DisentanglementDataset from pre-loaded data
    # Call train() function
    # Return checkpoint path
```

**Key Changes**:
- Accepts pre-loaded data instead of HDF5 path
- Creates dataset from numpy arrays
- Returns checkpoint path for later use

---

### 3. `eval_utils.py` (NEW)

**Purpose**: Reusable evaluation functions

**Functions**:
```python
def evaluate_single_dataset(
    embeddings: np.ndarray,
    labels: np.ndarray,
    content_groups: np.ndarray,
    checkpoint_path: str,
    input_dim: int = 768,
    output_dim: int = 128,
    batch_size: int = 256,
    device: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on single dataset (training set).
    
    Args:
        embeddings: Input embeddings
        labels: Labels
        content_groups: Content groups
        checkpoint_path: Path to model checkpoint
        ... (other args)
    
    Returns:
        {
            'input_metrics': {...},
            'model_metrics': {...}
        }
    """
    # Load model from checkpoint
    # Run inference
    # Compute metrics on input and model outputs
    # Return metrics dict

def evaluate_cross_dataset(
    train_real_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    checkpoint_path: str,
    input_dim: int = 768,
    output_dim: int = 128,
    batch_size: int = 256,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate cross-dataset generalization.
    
    Args:
        train_real_embeddings: Training real samples only
        test_embeddings: Test samples (all fake for Sora2)
        checkpoint_path: Path to model checkpoint
        ... (other args)
    
    Returns:
        {
            'cross_dataset_ami_input': ...,
            'cross_dataset_ami_z_auth': ...,
            ...
        }
    """
    # Load model
    # Run inference on both datasets
    # Compute cross-dataset metrics
    # Return metrics dict
```

**Key Changes**:
- Accepts pre-loaded data instead of HDF5 paths
- Returns metrics dicts (no printing, that's pipeline's job)
- Reusable by existing scripts

---

### 4. `pipeline.py` (NEW)

**Purpose**: Unified training + evaluation pipeline

**Structure**:
```python
def main():
    parser = argparse.ArgumentParser(...)
    
    # Arguments
    parser.add_argument('--train-hdf5', ...)
    parser.add_argument('--test-hdf5', ...)
    parser.add_argument('--encoder-name', ...)
    parser.add_argument('--output-dir', ...)  # Where to save everything
    parser.add_argument('--run-hyperparameter-sweep', action='store_true')
    
    args = parser.parse_args()
    
    # 1. Load datasets (once)
    print("Loading datasets...")
    train_embeddings, train_labels, train_content_groups = data_utils.load_data_from_hdf5(
        args.train_hdf5, args.encoder_name
    )
    test_embeddings, test_labels, test_content_groups = data_utils.load_data_from_hdf5(
        args.test_hdf5, args.encoder_name
    )
    
    # Separate train real for cross-dataset eval
    train_real_mask = train_labels == 0
    train_real_embeddings = train_embeddings[train_real_mask]
    
    # 2. Define hyperparameter configs
    configs = {
        'conservative': {
            'min_variance': 0.1,
            'variance_reg_weight': 1.0,
            'lambda_var': 0.5,
            'lambda_orth': 0.1,
            ...
        },
        'moderate': {
            'min_variance': 0.2,
            'variance_reg_weight': 2.0,
            'lambda_var': 0.5,
            'lambda_orth': 0.1,
            ...
        },
        'aggressive': {
            'min_variance': 0.5,
            'variance_reg_weight': 5.0,
            'lambda_var': 0.5,
            'lambda_orth': 0.1,
            ...
        }
    }
    
    # 3. Loop over configs
    all_results = {}
    
    for config_name, config in configs.items():
        print(f"\n{'='*80}")
        print(f"CONFIG: {config_name.upper()}")
        print(f"{'='*80}")
        
        # Create save directory for this config
        config_save_dir = os.path.join(args.output_dir, config_name)
        os.makedirs(config_save_dir, exist_ok=True)
        
        # 3a. Train model
        checkpoint_path = train_utils.train_model(
            train_embeddings,
            train_labels,
            train_content_groups,
            config,
            config_save_dir,
            device
        )
        
        # 3b. Evaluate on training set
        train_results = eval_utils.evaluate_single_dataset(
            train_embeddings,
            train_labels,
            train_content_groups,
            checkpoint_path,
            ...
        )
        
        # 3c. Evaluate on test set (cross-dataset)
        cross_dataset_results = eval_utils.evaluate_cross_dataset(
            train_real_embeddings,
            test_embeddings,
            checkpoint_path,
            ...
        )
        
        # Store results
        all_results[config_name] = {
            'config': config,
            'checkpoint_path': checkpoint_path,
            'train_metrics': train_results,
            'cross_dataset_metrics': cross_dataset_results
        }
        
        # Save individual config results
        with open(os.path.join(config_save_dir, 'results.json'), 'w') as f:
            json.dump(all_results[config_name], f, indent=2)
    
    # 4. Save combined results
    with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 5. Print summary comparison
    print_summary_table(all_results)
```

---

## Refactoring Steps

### Step 1: Create `data_utils.py`
- Extract `load_data_from_hdf5()` from `evaluate_metrics.py`
- Make it standalone and reusable
- Update `evaluate_metrics.py` to import from `data_utils`

### Step 2: Create `train_utils.py`
- Extract training logic from `main.py`
- Create function that accepts pre-loaded data
- Create helper to convert numpy arrays to dataset
- Return checkpoint path

### Step 3: Create `eval_utils.py`
- Extract evaluation logic from `evaluate_metrics.py`
- Extract cross-dataset logic from `evaluate_cross_dataset.py`
- Make functions accept pre-loaded data
- Return metrics dicts (no printing)

### Step 4: Create `pipeline.py`
- Implement main pipeline flow
- Load datasets once
- Loop over hyperparameter configs
- Call train_utils and eval_utils
- Save and print results

### Step 5: Update existing scripts (optional)
- `main.py`: Can import from `train_utils` (backward compatible)
- `evaluate_metrics.py`: Can import from `eval_utils` (backward compatible)
- `evaluate_cross_dataset.py`: Can import from `eval_utils` (backward compatible)

---

## Hyperparameter Configs

```python
HYPERPARAMETER_CONFIGS = {
    'conservative': {
        'min_variance': 0.1,
        'variance_reg_weight': 1.0,
        'lambda_var': 0.5,
        'lambda_orth': 0.1,
        'temperature': 0.1,
        'min_orth': 0.001,
        # No lambda_sep (anomaly detection approach)
    },
    'moderate': {
        'min_variance': 0.2,
        'variance_reg_weight': 2.0,
        'lambda_var': 0.5,
        'lambda_orth': 0.1,
        'temperature': 0.1,
        'min_orth': 0.001,
    },
    'aggressive': {
        'min_variance': 0.5,
        'variance_reg_weight': 5.0,
        'lambda_var': 0.5,
        'lambda_orth': 0.1,
        'temperature': 0.1,
        'min_orth': 0.001,
    }
}
```

---

## Usage

### Basic (single config):
```bash
python3 -m training.disentangled.pipeline \
    --train-hdf5 exports/deepfake_embeddings_2.h5 \
    --test-hdf5 exports/sora2_embeddings.h5 \
    --encoder-name hubert \
    --output-dir results/pipeline_run_001
```

### With hyperparameter sweep:
```bash
python3 -m training.disentangled.pipeline \
    --train-hdf5 exports/deepfake_embeddings_2.h5 \
    --test-hdf5 exports/sora2_embeddings.h5 \
    --encoder-name hubert \
    --output-dir results/pipeline_sweep_001 \
    --run-hyperparameter-sweep
```

---

## Output Structure

```
results/disentangled/pipeline_sweep_001/
├── all_results.json              # Combined results for all configs
├── conservative/
│   ├── best_model.pt            # Trained model
│   ├── metrics_history.json     # Training history
│   └── results.json             # Evaluation results
├── moderate/
│   ├── best_model.pt
│   ├── metrics_history.json
│   └── results.json
└── aggressive/
    ├── best_model.pt
    ├── metrics_history.json
    └── results.json
```

---

## Benefits

1. **No code duplication**: Data loading and evaluation logic shared
2. **Modular**: Each component can be used independently
3. **Easy to extend**: Add new configs or evaluation metrics easily
4. **Backward compatible**: Existing scripts still work
5. **Clean separation**: Data loading → Training → Evaluation
6. **Efficient**: Load datasets once, reuse for all configs

---

## Implementation Order

1. ✅ Create `data_utils.py` (extract from evaluate_metrics.py)
2. ✅ Create `train_utils.py` (extract from main.py)
3. ✅ Create `eval_utils.py` (extract from evaluate_metrics.py and evaluate_cross_dataset.py)
4. ✅ Create `pipeline.py` (orchestrate everything)
5. ✅ Test pipeline with single config
6. ✅ Test hyperparameter sweep
7. ✅ Update existing scripts to use shared utils (optional, for backward compat)

---

## Key Design Decisions

1. **No separation loss**: User wants anomaly detection approach (real-only training)
2. **Pre-load data**: Load once, reuse for training and evaluation
3. **Return dicts, don't print**: Evaluation functions return metrics, pipeline handles printing
4. **Config-based**: Hyperparameters in dicts, easy to add new configs
5. **Save everything**: Each config gets its own directory with model and results

