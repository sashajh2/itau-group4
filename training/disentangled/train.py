"""
Training and validation loops for disentangled representation learning.
"""
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict
import os
import json

from training.disentangled.losses import compute_total_loss
from training.disentangled.model import DisentangledProjector
from training.disentangled.metrics import evaluate_embeddings


def train_epoch(
    model: DisentangledProjector,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_var: float = 0.5,
    lambda_orth: float = 0.1,
    temperature: float = 0.1,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: DisentangledProjector
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        lambda_var: Weight for variance loss
        lambda_orth: Weight for orthogonality loss
        temperature: Temperature for prototypical loss
    
    Returns:
        Average losses for the epoch
    """
    model.train()
    
    epoch_losses = {
        'total': 0.0,
        'proto': 0.0,
        'var': 0.0,
        'orth': 0.0,
    }
    
    num_batches = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Move to device
        embeddings = batch['embeddings'].to(device)
        is_real = batch['is_real'].to(device)
        content_groups = batch['content_groups'].to(device)
        
        # Forward pass through both projection heads
        z_auth, z_id = model(embeddings)
        
        # Compute losses
        total_loss, losses_dict = compute_total_loss(
            z_id, z_auth, is_real, content_groups,
            lambda_var=lambda_var,
            lambda_orth=lambda_orth,
            temperature=temperature,
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        for key in epoch_losses:
            epoch_losses[key] += losses_dict[key]
        num_batches += 1
    
    # Average losses
    if num_batches > 0:
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
    
    return epoch_losses


def validate_epoch(
    model: DisentangledProjector,
    dataloader: DataLoader,
    device: torch.device,
    lambda_var: float = 0.5,
    lambda_orth: float = 0.1,
    temperature: float = 0.1,
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        model: DisentangledProjector
        dataloader: Validation dataloader
        device: Device to validate on
        lambda_var: Weight for variance loss
        lambda_orth: Weight for orthogonality loss
        temperature: Temperature for prototypical loss
    
    Returns:
        Average losses for the epoch
    """
    model.eval()
    
    epoch_losses = {
        'total': 0.0,
        'proto': 0.0,
        'var': 0.0,
        'orth': 0.0,
    }
    
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            embeddings = batch['embeddings'].to(device)
            is_real = batch['is_real'].to(device)
            content_groups = batch['content_groups'].to(device)
            
            z_auth, z_id = model(embeddings)
            
            total_loss, losses_dict = compute_total_loss(
                z_id, z_auth, is_real, content_groups,
                lambda_var=lambda_var,
                lambda_orth=lambda_orth,
                temperature=temperature,
            )
            
            for key in epoch_losses:
                epoch_losses[key] += losses_dict[key]
            num_batches += 1
    
    # Average losses
    if num_batches > 0:
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
    
    return epoch_losses


def train(
    model: DisentangledProjector,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-4,
    device: str = 'cuda',
    save_dir: str = './checkpoints',
    lambda_var: float = 0.5,
    lambda_orth: float = 0.1,
    temperature: float = 0.1,
) -> None:
    """
    Full training loop.
    
    Args:
        model: DisentangledProjector
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs: Number of epochs
        lr: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
        lambda_var: Weight for variance loss
        lambda_orth: Weight for orthogonality loss
        temperature: Temperature for prototypical loss
    """
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
    
    print(f"\n🚀 Starting training...")
    print(f"   Device: {device}")
    print(f"   Learning rate: {lr}")
    print(f"   Lambda var: {lambda_var}, Lambda orth: {lambda_orth}")
    print(f"   Temperature: {temperature}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Checkpoint dir: {save_dir}\n")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, device,
            lambda_var=lambda_var,
            lambda_orth=lambda_orth,
            temperature=temperature,
        )
        print(f"Train - Total: {train_losses['total']:.4f}, "
              f"Proto: {train_losses['proto']:.4f}, "
              f"Var: {train_losses['var']:.4f}, "
              f"Orth: {train_losses['orth']:.4f}")
        
        # Validate
        val_losses = validate_epoch(
            model, val_loader, device,
            lambda_var=lambda_var,
            lambda_orth=lambda_orth,
            temperature=temperature,
        )
        print(f"Val   - Total: {val_losses['total']:.4f}, "
              f"Proto: {val_losses['proto']:.4f}, "
              f"Var: {val_losses['var']:.4f}, "
              f"Orth: {val_losses['orth']:.4f}")
        
        # ============================================================
        # EVALUATE AFTER EPOCH (z^auth embeddings)
        # ============================================================
        print("\n📊 Evaluating z^auth embeddings...")
        
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
            checkpoint_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'metrics': after_metrics,
                'lambda_var': lambda_var,
                'lambda_orth': lambda_orth,
                'temperature': temperature,
            }, checkpoint_path)
            print(f"✅ Saved best model to {checkpoint_path} (val_loss: {best_val_loss:.4f})")
        
        # Save latest checkpoint
        latest_path = os.path.join(save_dir, 'latest_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metrics': after_metrics,
        }, latest_path)
        
        # Save metrics history
        metrics_path = os.path.join(save_dir, 'metrics_history.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    
    if len(metrics_history['epochs']) > 0:
        final_metrics = metrics_history['epochs'][-1]['metrics']
        print(f"\n📈 Final Improvement:")
        print(f"  AMI:             {before_metrics['ami']:.4f} → {final_metrics['ami']:.4f} "
              f"(Δ: {final_metrics['ami'] - before_metrics['ami']:+.4f})")
        print(f"  ARI:             {before_metrics['ari']:.4f} → {final_metrics['ari']:.4f} "
              f"(Δ: {final_metrics['ari'] - before_metrics['ari']:+.4f})")
        print(f"  Silhouette (GT): {before_metrics['silhouette_gt']:.4f} → {final_metrics['silhouette_gt']:.4f} "
              f"(Δ: {final_metrics['silhouette_gt'] - before_metrics['silhouette_gt']:+.4f})")
        print(f"  Silhouette (KM): {before_metrics['silhouette_clusters']:.4f} → {final_metrics['silhouette_clusters']:.4f} "
              f"(Δ: {final_metrics['silhouette_clusters'] - before_metrics['silhouette_clusters']:+.4f})")
        print(f"{'='*60}\n")
    
    print(f"🎉 Training completed!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Metrics history saved to: {os.path.join(save_dir, 'metrics_history.json')}")

