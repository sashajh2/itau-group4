"""
Training utilities for disentangled representation learning.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple

from training.disentangled.model import DisentangledProjector
from training.disentangled.dataset import disentanglement_collate_fn
from training.disentangled.train import train


class NumpyDataset(Dataset):
    """
    Dataset that works with pre-loaded numpy arrays.
    Used when data is already loaded in memory.
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        content_groups: np.ndarray,
    ):
        """
        Args:
            embeddings: [n_samples, emb_dim]
            labels: [n_samples] (0=real, 1=fake)
            content_groups: [n_samples] (content group IDs)
        """
        self.embeddings = embeddings
        self.labels = labels
        self.content_groups = content_groups
        
        # Convert labels to is_real boolean
        self.is_real = (labels == 0)
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Dict:
        return {
            'embedding': torch.from_numpy(self.embeddings[idx]).float(),
            'is_real': torch.tensor(self.is_real[idx], dtype=torch.bool),
            'content_group': self.content_groups[idx],
        }


def train_model(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    train_content_groups: np.ndarray,
    config: Dict,
    save_dir: str,
    input_dim: int = 768,
    output_dim: int = 128,
    batch_size: int = 128,
    num_epochs: int = 50,
    lr: float = 1e-4,
    num_workers: int = 4,
    val_split: float = 0.1,
    device: str = 'cuda',
    use_bn: bool = False,
    use_warmup: bool = True,
    warmup_steps: int = 1000,
    weight_decay: float = 1e-5,
    use_adamw: bool = True,
    gradient_clip: float = 1.0,
) -> str:
    """
    Train disentanglement model on pre-loaded data.
    
    Args:
        train_embeddings: Pre-loaded embeddings [n_samples, emb_dim]
        train_labels: Pre-loaded labels [n_samples] (0=real, 1=fake)
        train_content_groups: Pre-loaded content groups [n_samples]
        config: Dict with hyperparameters:
            - min_variance: float
            - variance_reg_weight: float
            - lambda_var: float
            - lambda_orth: float
            - temperature: float
            - min_orth: float
            - use_equal_weight_normalization: bool
            - use_adaptive_scaling: bool
            - min_loss_ratio: float
        save_dir: Directory to save checkpoint
        input_dim: Input embedding dimension
        output_dim: Output projection dimension
        batch_size: Batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        num_workers: Number of data loading workers
        val_split: Validation split ratio
        device: Device to train on
        use_bn: Use batch normalization
        use_warmup: Use learning rate warmup
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay for AdamW
        use_adamw: Use AdamW optimizer
        gradient_clip: Gradient clipping max norm
    
    Returns:
        checkpoint_path: Path to saved model checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create dataset from numpy arrays
    full_dataset = NumpyDataset(train_embeddings, train_labels, train_content_groups)
    
    # Split into train/val
    val_size = int(val_split * len(full_dataset))
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
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=disentanglement_collate_fn,
        pin_memory=True if device == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=disentanglement_collate_fn,
        pin_memory=True if device == 'cuda' else False,
    )
    
    # Create model
    print("🏗️  Creating model...")
    model = DisentangledProjector(
        input_dim=input_dim,
        output_dim=output_dim,
        use_bn=use_bn
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
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        save_dir=save_dir,
        lambda_var=config.get('lambda_var', 0.5),
        lambda_orth=config.get('lambda_orth', 0.1),
        temperature=config.get('temperature', 0.1),
        min_variance=config.get('min_variance', 0.01),
        variance_reg_weight=config.get('variance_reg_weight', 0.1),
        min_orth=config.get('min_orth', 0.001),
        use_adaptive_scaling=config.get('use_adaptive_scaling', False),
        use_equal_weight_normalization=config.get('use_equal_weight_normalization', True),
        min_loss_ratio=config.get('min_loss_ratio', 0.1),
        use_warmup=use_warmup,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        use_adamw=use_adamw,
        gradient_clip=gradient_clip,
    )
    
    checkpoint_path = os.path.join(save_dir, 'best_model.pt')
    return checkpoint_path

