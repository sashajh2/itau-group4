# Cursor Prompt: Add Evaluation Metrics to Training Loop

## Context

I need to add evaluation metrics to my disentangled representation learning training loop. These metrics should be computed:
1. **Before training starts** (on the input embeddings, pre-projection)
2. **After each epoch** (on the z^auth embeddings from f_auth projection head)

This will help me track whether the disentanglement is improving the embedding space quality.

## Metrics to Implement (Start with 3)

Based on Section 3.4.3 of my thesis (attached pages), I want to start with these three clustering-based metrics:

### 1. Adjusted Mutual Information (AMI)
- Equation 3.11 in thesis
- Measures agreement between k-means clusters (k=2) and ground truth labels
- Range: [0, 1], higher is better
- Implementation: `sklearn.metrics.adjusted_mutual_info_score`

### 2. Adjusted Rand Index (ARI)  
- Equation 3.13 in thesis
- Measures pairwise agreement between clustering and ground truth
- Range: [-1, 1], typically [0, 1], higher is better
- Implementation: `sklearn.metrics.adjusted_rand_score`

### 3. Silhouette Coefficient
- Equation 3.14 in thesis
- Measures cluster compactness and separation
- Range: [-1, 1], higher is better
- Use **cosine distance** (not Euclidean) since embeddings are normalized
- Implementation: `sklearn.metrics.silhouette_score`

## Requirements

### 1. Create Evaluation Module (`metrics.py`)

```python
"""
Evaluation metrics for disentangled representation learning.
Implements clustering-based metrics from Section 3.4.3.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score
)

def compute_clustering_metrics(embeddings, labels, metric='cosine', random_state=42):
    """
    Compute AMI, ARI, and Silhouette scores for embeddings.
    
    Args:
        embeddings: np.ndarray or torch.Tensor, shape [n_samples, emb_dim]
        labels: np.ndarray or torch.Tensor, shape [n_samples], binary 0=real, 1=fake
        metric: Distance metric for silhouette ('cosine' or 'euclidean')
        random_state: Random seed for k-means
    
    Returns:
        dict with keys: 'ami', 'ari', 'silhouette_gt', 'silhouette_clusters'
    """
    # Convert to numpy if needed
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Ensure labels are binary integers
    labels = labels.astype(int)
    
    # K-means clustering with k=2
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=random_state)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Compute metrics
    ami = adjusted_mutual_info_score(labels, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)
    
    # Silhouette with ground truth labels (primary metric)
    silhouette_gt = silhouette_score(embeddings, labels, metric=metric)
    
    # Silhouette with cluster labels (validation metric)
    silhouette_clusters = silhouette_score(embeddings, cluster_labels, metric=metric)
    
    return {
        'ami': ami,
        'ari': ari,
        'silhouette_gt': silhouette_gt,
        'silhouette_clusters': silhouette_clusters,
    }


def evaluate_embeddings(model, dataloader, device, use_auth=True, max_samples=10000):
    """
    Evaluate embeddings from model on dataset.
    
    Args:
        model: DisentangledProjector model
        dataloader: DataLoader for evaluation
        device: Device to run on
        use_auth: If True, use z^auth; if False, use input embeddings
        max_samples: Maximum samples to evaluate (for speed)
    
    Returns:
        dict of metrics
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            embeddings = batch['embeddings'].to(device)
            is_real = batch['is_real'].to(device)
            
            if use_auth:
                # Use z^auth from projection head
                z_auth, _ = model(embeddings)
                all_embeddings.append(z_auth.cpu())
            else:
                # Use input embeddings (before projection)
                all_embeddings.append(embeddings.cpu())
            
            # Convert is_real to 0/1 labels
            labels = (~is_real).long()  # 0=real, 1=fake
            all_labels.append(labels.cpu())
            
            # Limit samples for speed
            if len(all_embeddings) * embeddings.shape[0] >= max_samples:
                break
    
    # Concatenate all batches
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Compute metrics
    metrics = compute_clustering_metrics(all_embeddings, all_labels, metric='cosine')
    
    return metrics
```

### 2. Modify Training Loop (`train.py`)

Add evaluation calls to the training loop:

```python
def train(model, train_loader, val_loader, num_epochs=50, lr=1e-4, 
          device='cuda', save_dir='./checkpoints'):
    """
    Training loop with metric evaluation.
    """
    import os
    import json
    from datetime import datetime
    
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    
    # Track metrics over time
    metrics_history = {
        'before_training': {},
        'epochs': [],
    }
    
    # ============================================================
    # EVALUATE BEFORE TRAINING (Input embeddings, no projection)
    # ============================================================
    print("\n" + "="*60)
    print("EVALUATING INPUT EMBEDDINGS (Before Training)")
    print("="*60)
    
    from metrics import evaluate_embeddings
    
    before_metrics = evaluate_embeddings(
        model, val_loader, device, 
        use_auth=False,  # Use input embeddings
        max_samples=10000
    )
    
    metrics_history['before_training'] = before_metrics
    
    print(f"Before Training Metrics:")
    print(f"  AMI:                {before_metrics['ami']:.4f}")
    print(f"  ARI:                {before_metrics['ari']:.4f}")
    print(f"  Silhouette (GT):    {before_metrics['silhouette_gt']:.4f}")
    print(f"  Silhouette (Kmeans):{before_metrics['silhouette_clusters']:.4f}")
    print("="*60 + "\n")
    
    best_val_loss = float('inf')
    
    # ============================================================
    # TRAINING LOOP
    # ============================================================
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device)
        print(f"Train - Total: {train_losses['total']:.4f}, "
              f"Proto: {train_losses['proto']:.4f}, "
              f"Var: {train_losses['var']:.4f}, "
              f"Orth: {train_losses['orth']:.4f}")
        
        # Validate
        val_losses = validate_epoch(model, val_loader, device)
        print(f"Val   - Total: {val_losses['total']:.4f}, "
              f"Proto: {val_losses['proto']:.4f}, "
              f"Var: {val_losses['var']:.4f}, "
              f"Orth: {val_losses['orth']:.4f}")
        
        # ============================================================
        # EVALUATE AFTER EPOCH (z^auth embeddings)
        # ============================================================
        print("\nEvaluating z^auth embeddings...")
        
        after_metrics = evaluate_embeddings(
            model, val_loader, device,
            use_auth=True,  # Use z^auth from f_auth projection
            max_samples=10000
        )
        
        print(f"After Epoch {epoch+1} Metrics:")
        print(f"  AMI:                {after_metrics['ami']:.4f} "
              f"(Δ: {after_metrics['ami'] - before_metrics['ami']:+.4f})")
        print(f"  ARI:                {after_metrics['ari']:.4f} "
              f"(Δ: {after_metrics['ari'] - before_metrics['ari']:+.4f})")
        print(f"  Silhouette (GT):    {after_metrics['silhouette_gt']:.4f} "
              f"(Δ: {after_metrics['silhouette_gt'] - before_metrics['silhouette_gt']:+.4f})")
        print(f"  Silhouette (Kmeans):{after_metrics['silhouette_clusters']:.4f} "
              f"(Δ: {after_metrics['silhouette_clusters'] - before_metrics['silhouette_clusters']:+.4f})")
        
        # Store in history
        epoch_record = {
            'epoch': epoch + 1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metrics': after_metrics,
        }
        metrics_history['epochs'].append(epoch_record)
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'metrics': after_metrics,
            }, os.path.join(save_dir, 'best_model.pt'))
            print("✓ Saved best model!")
        
        # Save metrics history
        with open(os.path.join(save_dir, 'metrics_history.json'), 'w') as f:
            json.dump(metrics_history, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nFinal Improvement:")
    final_metrics = metrics_history['epochs'][-1]['metrics']
    print(f"  AMI:             {before_metrics['ami']:.4f} → {final_metrics['ami']:.4f} "
          f"(Δ: {final_metrics['ami'] - before_metrics['ami']:+.4f})")
    print(f"  ARI:             {before_metrics['ari']:.4f} → {final_metrics['ari']:.4f} "
          f"(Δ: {final_metrics['ari'] - before_metrics['ari']:+.4f})")
    print(f"  Silhouette (GT): {before_metrics['silhouette_gt']:.4f} → {final_metrics['silhouette_gt']:.4f} "
          f"(Δ: {final_metrics['silhouette_gt'] - before_metrics['silhouette_gt']:+.4f})")
    print("="*60 + "\n")
```

### 3. Create Visualization Script (`plot_metrics.py`)

```python
"""
Plot metrics over training.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics_history(metrics_path='./checkpoints/metrics_history.json', 
                        output_path='./checkpoints/metrics_plot.png'):
    """
    Plot AMI, ARI, and Silhouette over training epochs.
    """
    with open(metrics_path, 'r') as f:
        history = json.load(f)
    
    before = history['before_training']
    epochs_data = history['epochs']
    
    # Extract metrics per epoch
    epochs = [e['epoch'] for e in epochs_data]
    ami = [e['metrics']['ami'] for e in epochs_data]
    ari = [e['metrics']['ari'] for e in epochs_data]
    sil_gt = [e['metrics']['silhouette_gt'] for e in epochs_data]
    sil_km = [e['metrics']['silhouette_clusters'] for e in epochs_data]
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Embedding Quality Metrics Over Training', fontsize=16, fontweight='bold')
    
    # AMI
    ax = axes[0, 0]
    ax.axhline(before['ami'], color='red', linestyle='--', alpha=0.5, label='Before Training')
    ax.plot(epochs, ami, marker='o', linewidth=2, markersize=4, label='After Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AMI')
    ax.set_title('Adjusted Mutual Information')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # ARI
    ax = axes[0, 1]
    ax.axhline(before['ari'], color='red', linestyle='--', alpha=0.5, label='Before Training')
    ax.plot(epochs, ari, marker='o', linewidth=2, markersize=4, label='After Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ARI')
    ax.set_title('Adjusted Rand Index')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Silhouette (Ground Truth)
    ax = axes[1, 0]
    ax.axhline(before['silhouette_gt'], color='red', linestyle='--', alpha=0.5, label='Before Training')
    ax.plot(epochs, sil_gt, marker='o', linewidth=2, markersize=4, label='After Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Coefficient (Ground Truth Labels)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1, 1])
    
    # Silhouette (K-means)
    ax = axes[1, 1]
    ax.axhline(before['silhouette_clusters'], color='red', linestyle='--', alpha=0.5, label='Before Training')
    ax.plot(epochs, sil_km, marker='o', linewidth=2, markersize=4, label='After Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Coefficient (K-means Clusters)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics plot to {output_path}")
    plt.close()

if __name__ == '__main__':
    plot_metrics_history()
```

## What I Need

Please implement the following:

1. **Create `metrics.py`** with:
   - `compute_clustering_metrics()` function
   - `evaluate_embeddings()` function
   - Proper handling of torch tensors and numpy arrays
   - Use cosine distance for silhouette

2. **Modify `train.py`** to:
   - Import the metrics module
   - Evaluate input embeddings before training starts
   - Evaluate z^auth embeddings after each epoch
   - Print metrics with delta (Δ) showing improvement
   - Save metrics history to JSON file

3. **Create `plot_metrics.py`** to:
   - Load metrics history from JSON
   - Plot all 4 metrics over epochs
   - Show "before training" baseline as horizontal line
   - Save high-quality figure

4. **Update `requirements.txt`** to include:
   ```
   scikit-learn
   matplotlib
   ```

## Expected Behavior

When I run training, I should see:

```
============================================================
EVALUATING INPUT EMBEDDINGS (Before Training)
============================================================
Before Training Metrics:
  AMI:                0.0234
  ARI:                0.0156
  Silhouette (GT):    0.1234
  Silhouette (Kmeans):0.0987
============================================================

Epoch 1/50
Train - Total: 2.3456, Proto: 1.2345, Var: 0.8901, Orth: 0.2210
Val   - Total: 2.4567, Proto: 1.3456, Var: 0.8901, Orth: 0.2210

Evaluating z^auth embeddings...
After Epoch 1 Metrics:
  AMI:                0.0345 (Δ: +0.0111)
  ARI:                0.0278 (Δ: +0.0122)
  Silhouette (GT):    0.1567 (Δ: +0.0333)
  Silhouette (Kmeans):0.1234 (Δ: +0.0247)
✓ Saved best model!

Epoch 2/50
...
```

## Important Notes

1. **Cosine Distance**: Must use `metric='cosine'` in silhouette_score since embeddings are L2-normalized

2. **Max Samples**: Use `max_samples=10000` to speed up evaluation (metrics are stable with 10k samples)

3. **Ground Truth vs Clusters**: We compute silhouette twice:
   - With ground truth labels (primary metric)
   - With k-means clusters (validation)

4. **Expected Results**: After training, we expect:
   - AMI: Should increase from ~0.02 to 0.7-1.0
   - ARI: Should increase from ~0.02 to 0.7-1.0  
   - Silhouette: Should increase from ~0.1 to 0.5-0.8

5. **JSON Format**: Save metrics as JSON for easy analysis and plotting later

## Files to Create/Modify

- [ ] `metrics.py` (new)
- [ ] `train.py` (modify)
- [ ] `plot_metrics.py` (new)
- [ ] `requirements.txt` (update)

Please implement this step-by-step, starting with the metrics module. Let me know if you need clarification on any of the equations or expected behavior!

## References

- Section 3.4.3: Representation Quality Analysis (attached thesis pages)
- Equation 3.11: AMI formula
- Equation 3.13: ARI formula
- Equation 3.14: Silhouette formula
