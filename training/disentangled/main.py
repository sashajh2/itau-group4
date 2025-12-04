"""
Main training script for disentangled representation learning.
"""
import argparse
import torch
from torch.utils.data import DataLoader

from training.disentangled.model import DisentangledProjector
from training.disentangled.dataset import DisentanglementDataset, disentanglement_collate_fn
from training.disentangled.train import train


def create_dataloader(
    hdf5_path: str,
    encoder_name: str = 'hubert',
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create DataLoader for training/validation.
    
    Args:
        hdf5_path: Path to HDF5 file
        encoder_name: Encoder to use
        batch_size: Batch size (recommend 128-256 for segment-level samples)
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader
    """
    dataset = DisentanglementDataset(hdf5_path, encoder_name)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=disentanglement_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    return dataloader


def main():
    parser = argparse.ArgumentParser(description="Train disentangled representation learning model")
    
    # Data arguments
    parser.add_argument('--hdf5-path', type=str, required=True,
                       help='Path to HDF5 file with embeddings')
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
    
    # Loss hyperparameters
    parser.add_argument('--lambda-var', type=float, default=0.5,
                       help='Weight for variance loss (default: 0.5)')
    parser.add_argument('--lambda-orth', type=float, default=0.1,
                       help='Weight for orthogonality loss (default: 0.1)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for prototypical loss (default: 0.1)')
    parser.add_argument('--min-variance', type=float, default=0.01,
                       help='Minimum variance threshold for regularization (default: 0.01)')
    parser.add_argument('--variance-reg-weight', type=float, default=0.1,
                       help='Weight for variance regularization term (default: 0.1)')
    parser.add_argument('--min-orth', type=float, default=0.001,
                       help='Minimum orthogonality loss value to prevent collapse (default: 0.001)')
    parser.add_argument('--use-equal-weight-normalization', action='store_true', default=True,
                       help='Use equal-weight normalization (normalize by initial values, default: True)')
    parser.add_argument('--no-equal-weight-normalization', dest='use_equal_weight_normalization', action='store_false',
                       help='Disable equal-weight normalization')
    parser.add_argument('--use-adaptive-scaling', action='store_true',
                       help='Use adaptive loss scaling (alternative to equal-weight normalization)')
    parser.add_argument('--no-adaptive-scaling', dest='use_adaptive_scaling', action='store_false',
                       help='Disable adaptive loss scaling')
    parser.set_defaults(use_adaptive_scaling=False, use_equal_weight_normalization=True)
    parser.add_argument('--min-loss-ratio', type=float, default=0.1,
                       help='Minimum ratio of each loss to max loss (prevents collapse, default: 0.1)')
    
    # Other arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints/disentangled',
                       help='Directory to save checkpoints (default: ./checkpoints/disentangled)')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio (default: 0.1)')
    parser.add_argument('--min-samples-per-group', type=int, default=3,
                       help='Minimum samples per content group to include (default: 3)')
    
    # Training improvements
    parser.add_argument('--use-warmup', action='store_true', default=True,
                       help='Use learning rate warmup (default: True)')
    parser.add_argument('--no-warmup', dest='use_warmup', action='store_false',
                       help='Disable learning rate warmup')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Number of warmup steps (default: 1000)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay for AdamW (default: 1e-5)')
    parser.add_argument('--use-adamw', action='store_true', default=True,
                       help='Use AdamW optimizer (default: True)')
    parser.add_argument('--use-adam', dest='use_adamw', action='store_false',
                       help='Use Adam optimizer instead of AdamW')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping max norm (default: 1.0, set to 0 to disable)')
    parser.add_argument('--use-bn', action='store_true',
                       help='Use batch normalization in projection heads')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\n{'='*60}")
    print(f"Disentangled Representation Learning Training")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  HDF5 path: {args.hdf5_path}")
    print(f"  Encoder: {args.encoder_name}")
    print(f"  Input dim: {args.input_dim}")
    print(f"  Output dim: {args.output_dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Lambda var: {args.lambda_var}")
    print(f"  Lambda orth: {args.lambda_orth}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Min variance: {args.min_variance}")
    print(f"  Variance reg weight: {args.variance_reg_weight}")
    print(f"  Adaptive scaling: {args.use_adaptive_scaling}")
    print(f"  Min samples per group: {args.min_samples_per_group}")
    print(f"  Device: {device}")
    print(f"  Save dir: {args.save_dir}")
    print(f"{'='*60}\n")
    
    # Create full dataset
    print("📂 Loading dataset...")
    full_dataset = DisentanglementDataset(
        args.hdf5_path, 
        args.encoder_name,
        min_samples_per_group=args.min_samples_per_group
    )
    
    # Split into train/val
    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=disentanglement_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=disentanglement_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # Create model
    print("🏗️  Creating model...")
    model = DisentangledProjector(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        use_bn=args.use_bn
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")
    
    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir,
        lambda_var=args.lambda_var,
        lambda_orth=args.lambda_orth,
        temperature=args.temperature,
        min_variance=args.min_variance,
        variance_reg_weight=args.variance_reg_weight,
        min_orth=args.min_orth,
        use_adaptive_scaling=args.use_adaptive_scaling,
        use_equal_weight_normalization=args.use_equal_weight_normalization,
        min_loss_ratio=args.min_loss_ratio,
        use_warmup=args.use_warmup,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        use_adamw=args.use_adamw,
        gradient_clip=args.gradient_clip,
    )


if __name__ == '__main__':
    main()

