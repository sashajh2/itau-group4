"""
Quick training script for disentangled representation learning.
Skips initial evaluation to avoid potential segfaults.
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.disentangled.model import DisentangledProjector
from training.disentangled.dataset import DisentanglementDataset, disentanglement_collate_fn
from training.disentangled.losses import compute_total_loss, EqualWeightNormalizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import os


def main():
    # Configuration
    hdf5_path = "exports/deepfake_embeddings_2.h5"
    num_epochs = 1
    lambda_var = 0.0  # No variance loss
    lambda_orth = 0.1
    temperature = 0.1
    lr = 1e-4
    batch_size = 128
    save_dir = "./checkpoints/disentangled_no_var"
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"\n{'='*60}")
    print(f"Quick Training: lambda_var={lambda_var}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    full_dataset = DisentanglementDataset(hdf5_path, "hubert", min_samples_per_group=3)
    
    # Split into train/val (70/30)
    val_size = int(0.3 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=disentanglement_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=disentanglement_collate_fn
    )
    
    # Create model
    print("\n🏗️  Creating model...")
    model = DisentangledProjector(input_dim=768, output_dim=128)
    model = model.to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    warmup_steps = 1000
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Loss normalizer
    equal_weight_normalizer = EqualWeightNormalizer()
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n🚀 Starting training (lambda_var={lambda_var}, lambda_orth={lambda_orth})...")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {'total': 0.0, 'proto': 0.0, 'var': 0.0, 'orth': 0.0}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            embeddings = batch['embeddings'].to(device)
            is_real = batch['is_real'].to(device)
            content_groups = batch['content_groups'].to(device)
            
            # Forward pass
            z_auth, z_id = model(embeddings)
            
            # Compute losses
            total_loss, losses_dict = compute_total_loss(
                z_id, z_auth, is_real, content_groups,
                lambda_var=lambda_var,
                lambda_orth=lambda_orth,
                temperature=temperature,
                equal_weight_normalizer=equal_weight_normalizer,
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses_dict[key]
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses_dict['total']:.4f}",
                'proto': f"{losses_dict['proto']:.4f}",
                'orth': f"{losses_dict['orth']:.4f}"
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Validation
        model.eval()
        val_losses = {'total': 0.0, 'proto': 0.0, 'var': 0.0, 'orth': 0.0}
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embeddings'].to(device)
                is_real = batch['is_real'].to(device)
                content_groups = batch['content_groups'].to(device)
                
                z_auth, z_id = model(embeddings)
                total_loss, losses_dict = compute_total_loss(
                    z_id, z_auth, is_real, content_groups,
                    lambda_var=lambda_var,
                    lambda_orth=lambda_orth,
                    temperature=temperature,
                    equal_weight_normalizer=equal_weight_normalizer,
                )
                
                for key in val_losses:
                    val_losses[key] += losses_dict[key]
                val_batches += 1
        
        for key in val_losses:
            val_losses[key] /= val_batches
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train - Total: {epoch_losses['total']:.4f}, Proto: {epoch_losses['proto']:.4f}, Var: {epoch_losses['var']:.4f}, Orth: {epoch_losses['orth']:.4f}")
        print(f"  Val   - Total: {val_losses['total']:.4f}, Proto: {val_losses['proto']:.4f}, Var: {val_losses['var']:.4f}, Orth: {val_losses['orth']:.4f}")
    
    # Save final model
    checkpoint_path = os.path.join(save_dir, "final_model.pt")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': epoch_losses,
        'val_losses': val_losses,
        'config': {
            'lambda_var': lambda_var,
            'lambda_orth': lambda_orth,
            'temperature': temperature,
            'lr': lr,
        }
    }, checkpoint_path)
    print(f"\n✅ Model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
