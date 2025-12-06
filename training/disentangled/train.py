"""
Training and validation loops for disentangled representation learning.
"""
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional
import os
import json
import math

from training.disentangled.losses import compute_total_loss, AdaptiveLossScaler, EqualWeightNormalizer
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
    min_variance: float = 0.01,
    variance_reg_weight: float = 0.1,
    min_orth: float = 0.001,
    adaptive_scaler: Optional[AdaptiveLossScaler] = None,
    equal_weight_normalizer: Optional[EqualWeightNormalizer] = None,
    scheduler: Optional[LambdaLR] = None,
    gradient_clip: float = 1.0,
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
            min_variance=min_variance,
            variance_reg_weight=variance_reg_weight,
            min_orth=min_orth,
            adaptive_scaler=adaptive_scaler,
            equal_weight_normalizer=equal_weight_normalizer,
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        
        optimizer.step()
        
        # Update learning rate scheduler (for warmup) - step per batch
        if scheduler is not None:
            scheduler.step()
        
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
    min_variance: float = 0.01,
    variance_reg_weight: float = 0.1,
    min_orth: float = 0.001,
    adaptive_scaler: Optional[AdaptiveLossScaler] = None,
    equal_weight_normalizer: Optional[EqualWeightNormalizer] = None,
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
            min_variance=min_variance,
            variance_reg_weight=variance_reg_weight,
            min_orth=min_orth,
            adaptive_scaler=adaptive_scaler,
            equal_weight_normalizer=equal_weight_normalizer,
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
    min_variance: float = 0.01,
    variance_reg_weight: float = 0.1,
    min_orth: float = 0.001,
    use_adaptive_scaling: bool = False,
    use_equal_weight_normalization: bool = True,
    min_loss_ratio: float = 0.1,
    use_warmup: bool = True,
    warmup_steps: int = 1000,
    weight_decay: float = 1e-5,
    use_adamw: bool = True,
    gradient_clip: float = 1.0,
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
    
    # Use AdamW with weight decay for better generalization
    if use_adamw:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"🔧 Using AdamW optimizer (weight_decay={weight_decay})")
    else:
        optimizer = Adam(model.parameters(), lr=lr)
        print(f"🔧 Using Adam optimizer")
    
    # Learning rate warmup scheduler
    if use_warmup:
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps  # Linear warmup
            else:
                return 1.0  # Constant after warmup
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f"📈 Using learning rate warmup ({warmup_steps} steps)")
    else:
        scheduler = None
    
    # Initialize loss balancing
    if use_equal_weight_normalization:
        equal_weight_normalizer = EqualWeightNormalizer()
        adaptive_scaler = None
        print("⚖️  Using equal-weight normalization (normalize by initial values, then equal weights)")
    elif use_adaptive_scaling:
        adaptive_scaler = AdaptiveLossScaler(
            alpha=0.99, 
            warmup_steps=warmup_steps,
            min_loss_ratio=min_loss_ratio
        )
        equal_weight_normalizer = None
        print(f"📊 Using adaptive loss scaling (min_loss_ratio={min_loss_ratio})")
    else:
        adaptive_scaler = None
        equal_weight_normalizer = None
        print("⚠️  Using fixed loss weights (no balancing)")
    
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
        max_samples=10000,
        max_samples_for_silhouette=1000  # Use subset for fast silhouette during training
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
            min_variance=min_variance,
            variance_reg_weight=variance_reg_weight,
            adaptive_scaler=adaptive_scaler,
            scheduler=scheduler,
            gradient_clip=gradient_clip,
        )
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            print(f"   Current LR: {current_lr:.2e}")
        loss_str = f"Train - Total: {train_losses['total']:.4f}, "
        loss_str += f"Proto: {train_losses['proto']:.4f}, "
        loss_str += f"Var: {train_losses['var']:.4f}, "
        loss_str += f"Orth: {train_losses['orth']:.4f}"
        if 'proto_scaled' in train_losses:
            loss_str += f"\n        Scaled - Proto: {train_losses['proto_scaled']:.4f}, "
            loss_str += f"Var: {train_losses['var_scaled']:.4f}, "
            loss_str += f"Orth: {train_losses['orth_scaled']:.4f}"
        print(loss_str)
        
        # Validate
        val_losses = validate_epoch(
            model, val_loader, device,
            lambda_var=lambda_var,
            lambda_orth=lambda_orth,
            temperature=temperature,
            min_variance=min_variance,
            variance_reg_weight=variance_reg_weight,
            min_orth=min_orth,
            adaptive_scaler=adaptive_scaler,
        )
        loss_str = f"Val   - Total: {val_losses['total']:.4f}, "
        loss_str += f"Proto: {val_losses['proto']:.4f}, "
        loss_str += f"Var: {val_losses['var']:.4f}, "
        loss_str += f"Orth: {val_losses['orth']:.4f}"
        if 'proto_scaled' in val_losses:
            loss_str += f"\n        Scaled - Proto: {val_losses['proto_scaled']:.4f}, "
            loss_str += f"Var: {val_losses['var_scaled']:.4f}, "
            loss_str += f"Orth: {val_losses['orth_scaled']:.4f}"
        print(loss_str)
        
        # ============================================================
        # EVALUATE AFTER EPOCH (z^auth embeddings)
        # ============================================================
        print("\n📊 Evaluating z^auth embeddings...")
        
        after_metrics = evaluate_embeddings(
            model, val_loader, device,
            use_auth=True,  # Use z^auth from f_auth projection
            max_samples=10000,
            max_samples_for_silhouette=1000  # Use subset for fast silhouette during training
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

