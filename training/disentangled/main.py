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
    
    # Other arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints/disentangled',
                       help='Directory to save checkpoints (default: ./checkpoints/disentangled)')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio (default: 0.1)')
    
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
    print(f"  Device: {device}")
    print(f"  Save dir: {args.save_dir}")
    print(f"{'='*60}\n")
    
    # Create full dataset
    print("📂 Loading dataset...")
    full_dataset = DisentanglementDataset(args.hdf5_path, args.encoder_name)
    
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
        output_dim=args.output_dim
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
    )


if __name__ == '__main__':
    main()

