# Disentangled Representation Learning Implementation Guide

## Overview

This guide provides detailed instructions for implementing the disentangled representation learning approach described in Section 3.3.2 of your thesis. Each temporal segment will be treated as an independent sample, maximizing the amount of training data.

---

## Context

You're implementing a two-stage deepfake detection framework with:
- **Stage 1**: Disentangled representation learning (this guide)
- **Stage 2**: Temporal classification (separate)

Your HDF5 dataset contains:
- Pretrained embeddings at `/videos/{video_id}/embeddings/{encoder_name}/`
- Shape: `[num_augmentations, num_segments, embedding_dim]`
- Metadata: `source_idx` groups augmentations from the same source video
- Labels: `audio` and `video` labels for each augmentation

---

## Key Design Decisions

### 1. Temporal Segments as Individual Samples

**Decision**: Treat each temporal segment as a separate training sample.

**Rationale**:
- Maximizes training data (if you have 10 augmentations × 50 segments = 500 samples per source video)
- Allows the model to learn frame-level authenticity cues
- Maintains temporal granularity for later classification stage

**Implementation**:
```python
# Shape transformation:
# From: [num_augmentations, num_segments, emb_dim]
# To: [num_augmentations * num_segments, emb_dim]

# Each (augmentation_idx, segment_idx) becomes one sample
```

### 2. Content Grouping Strategy

**For Prototypical Contrastive Learning (z^id)**:

Use **(source_idx, segment_idx)** as the content group identifier.

**Rationale**:
- Segments at the same timestamp across augmentations have identical content
- Different augmentations of the same source at time t=5 should cluster together in z^id
- This leverages the temporal alignment in your dataset

**Example**:
```
Source Video A at t=5:
  - Augmentation 1, segment 5 → Content Group (A, 5)
  - Augmentation 2, segment 5 → Content Group (A, 5)
  - Augmentation 3, segment 5 → Content Group (A, 5)
  
Source Video A at t=6:
  - Augmentation 1, segment 6 → Content Group (A, 6)
  ...
```

### 3. Variance Minimization Strategy

**For Variance Minimization (z^auth)**:

Collect **ALL real segments** in the batch (regardless of source or timestamp).

**Rationale**:
- You want to learn identity-agnostic, content-agnostic authenticity cues
- All real video segments should cluster around one global "real manifold"
- This prevents the model from learning identity-specific or scene-specific patterns

**Implementation**:
```python
# In each batch:
real_mask = (audio_labels == 0) & (video_labels == 0)
z_auth_real = z_auth[real_mask]  # All real segments, any source/time

# Compute global real centroid
mu_real = z_auth_real.mean(dim=0)

# Minimize variance
L_var = ((z_auth_real - mu_real) ** 2).sum(dim=1).mean()
```

---

## Architecture

### Dual Projection Heads

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DisentangledProjector(nn.Module):
    """
    Dual projection heads for disentangling identity and authenticity.
    
    Args:
        input_dim: Dimension of input embeddings (e.g., 768 for many encoders)
        output_dim: Dimension of projected embeddings (default: 128)
    """
    def __init__(self, input_dim=768, output_dim=128):
        super().__init__()
        
        # Authenticity projection head
        self.f_auth = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
        # Identity projection head
        self.f_id = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, z):
        """
        Args:
            z: Input embeddings, shape [batch_size, input_dim]
        
        Returns:
            z_auth: Authenticity embeddings, shape [batch_size, output_dim]
            z_id: Identity embeddings, shape [batch_size, output_dim]
        """
        z_auth = F.normalize(self.f_auth(z), dim=-1)
        z_id = F.normalize(self.f_id(z), dim=-1)
        return z_auth, z_id
```

---

## Loss Functions

### Loss 1: Variance Minimization (Equation 3.6)

**Purpose**: Force real samples to cluster tightly in z^auth space.

**Implementation**:

```python
def variance_loss(z_auth, is_real):
    """
    Minimize variance of real samples around their centroid.
    
    Args:
        z_auth: Authenticity embeddings, shape [batch_size, emb_dim]
        is_real: Boolean mask, shape [batch_size]
    
    Returns:
        scalar loss
    """
    z_auth_real = z_auth[is_real]
    
    if z_auth_real.shape[0] < 2:
        return torch.tensor(0.0, device=z_auth.device)
    
    # Compute real centroid
    mu_real = z_auth_real.mean(dim=0, keepdim=True)
    
    # Minimize squared distances to centroid
    loss = ((z_auth_real - mu_real) ** 2).sum(dim=1).mean()
    
    return loss
```

**Key Points**:
- Only uses real samples (audio_label=0 AND video_label=0)
- Computes global centroid across all real segments in batch
- Encourages tight clustering (low variance)

---

### Loss 2: Prototypical Contrastive Loss (Equation 3.5)

**Purpose**: Cluster embeddings by content group in z^id space.

**Implementation**:

```python
def prototypical_contrastive_loss(z_id, content_groups, temperature=0.1):
    """
    Cluster samples by content group using prototypical contrastive learning.
    
    Args:
        z_id: Identity embeddings, shape [batch_size, emb_dim]
        content_groups: Content group IDs, shape [batch_size]
        temperature: Temperature hyperparameter (default: 0.1)
    
    Returns:
        scalar loss
    """
    device = z_id.device
    unique_groups = torch.unique(content_groups)
    
    # Compute prototypes for each content group
    prototypes = []
    prototype_labels = []
    
    for group_id in unique_groups:
        mask = content_groups == group_id
        group_embeddings = z_id[mask]
        prototype = group_embeddings.mean(dim=0)
        prototypes.append(prototype)
        prototype_labels.append(group_id)
    
    prototypes = torch.stack(prototypes)  # [num_groups, emb_dim]
    
    # Compute distances from each sample to all prototypes
    # Using negative Euclidean distance (higher = more similar)
    distances = -torch.cdist(z_id, prototypes, p=2)  # [batch_size, num_groups]
    
    # Create target indices (which prototype each sample belongs to)
    targets = torch.zeros(z_id.shape[0], dtype=torch.long, device=device)
    for i, group_id in enumerate(prototype_labels):
        targets[content_groups == group_id] = i
    
    # Apply temperature scaling and compute cross-entropy
    logits = distances / temperature
    loss = F.cross_entropy(logits, targets)
    
    return loss
```

**Key Points**:
- Groups samples by (source_idx, segment_idx)
- Computes one prototype per content group
- Uses Euclidean distance (as in Equation 3.5)
- Applies to ALL samples (real and fake)

---

### Loss 3: Orthogonality Constraint (Equation 3.3)

**Purpose**: Enforce independence between z^id and z^auth.

**Implementation**:

```python
def orthogonality_loss(z_id, z_auth):
    """
    Penalize correlation between identity and authenticity embeddings.
    
    Args:
        z_id: Identity embeddings, shape [batch_size, emb_dim]
        z_auth: Authenticity embeddings, shape [batch_size, emb_dim]
    
    Returns:
        scalar loss
    """
    batch_size = z_id.shape[0]
    
    # Compute pairwise cosine similarities
    # sim_matrix[i,j] = cos_sim(z_id[i], z_auth[j])
    sim_matrix = torch.matmul(z_id, z_auth.t())  # [batch_size, batch_size]
    
    # Average absolute similarity
    loss = sim_matrix.abs().sum() / (batch_size ** 2)
    
    return loss
```

**Key Points**:
- Measures correlation between z^id and z^auth across all sample pairs
- Uses cosine similarity (embeddings are already normalized)
- Should be close to zero if projections are orthogonal

---

### Total Loss (Equation 3.7)

```python
def compute_total_loss(z_id, z_auth, is_real, content_groups, 
                       lambda_var=0.5, lambda_orth=0.1):
    """
    Combine all three loss components.
    
    Args:
        z_id: Identity embeddings
        z_auth: Authenticity embeddings
        is_real: Boolean mask for real samples
        content_groups: Content group IDs
        lambda_var: Weight for variance loss (default: 0.5)
        lambda_orth: Weight for orthogonality loss (default: 0.1)
    
    Returns:
        total_loss, dict of individual losses
    """
    L_proto = prototypical_contrastive_loss(z_id, content_groups)
    L_var = variance_loss(z_auth, is_real)
    L_orth = orthogonality_loss(z_id, z_auth)
    
    total_loss = L_proto + lambda_var * L_var + lambda_orth * L_orth
    
    losses_dict = {
        'total': total_loss.item(),
        'proto': L_proto.item(),
        'var': L_var.item(),
        'orth': L_orth.item(),
    }
    
    return total_loss, losses_dict
```

---

## Dataset Implementation

### HDF5 Dataset Class

```python
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class DisentanglementDataset(Dataset):
    """
    Dataset for disentangled representation learning.
    Each temporal segment is treated as an independent sample.
    """
    
    def __init__(self, hdf5_path, encoder_name='hubert', split='train'):
        """
        Args:
            hdf5_path: Path to HDF5 file
            encoder_name: Which encoder embeddings to use ('hubert', 'openl3', 'senet')
            split: 'train', 'val', or 'test'
        """
        self.hdf5_path = hdf5_path
        self.encoder_name = encoder_name
        self.split = split
        
        # Build index of all samples
        self.samples = []
        
        with h5py.File(hdf5_path, 'r') as f:
            videos_group = f['/videos']
            
            for video_id in videos_group.keys():
                video = videos_group[video_id]
                
                # Load metadata
                source_idx = video['metadata'].attrs['source_idx']
                embeddings = video['embeddings'][encoder_name][:]  # [num_augs, num_segs, emb_dim]
                audio_labels = video['labels']['audio'][:]  # [num_augs]
                video_labels = video['labels']['video'][:]  # [num_augs]
                
                num_augs, num_segs, emb_dim = embeddings.shape
                
                # Create one sample per (augmentation, segment)
                for aug_idx in range(num_augs):
                    for seg_idx in range(num_segs):
                        sample = {
                            'video_id': video_id,
                            'aug_idx': aug_idx,
                            'seg_idx': seg_idx,
                            'source_idx': source_idx,
                            'audio_label': audio_labels[aug_idx],
                            'video_label': video_labels[aug_idx],
                        }
                        self.samples.append(sample)
        
        print(f"Loaded {len(self.samples)} samples from {hdf5_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load embedding from HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            video = f['/videos'][sample_info['video_id']]
            embedding = video['embeddings'][self.encoder_name][
                sample_info['aug_idx'], 
                sample_info['seg_idx']
            ]
        
        # Determine if sample is real (both audio and video are authentic)
        is_real = (sample_info['audio_label'] == 0) and (sample_info['video_label'] == 0)
        
        # Create content group ID: (source_idx, segment_idx)
        # This ensures segments at same timestamp across augmentations cluster together
        content_group = (sample_info['source_idx'], sample_info['seg_idx'])
        
        return {
            'embedding': torch.from_numpy(embedding).float(),
            'is_real': torch.tensor(is_real, dtype=torch.bool),
            'content_group': content_group,
            'source_idx': sample_info['source_idx'],
            'seg_idx': sample_info['seg_idx'],
        }
```

---

### Custom Collate Function

```python
def disentanglement_collate_fn(batch):
    """
    Custom collate function to create batches for disentangled learning.
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Dictionary with batched tensors
    """
    embeddings = torch.stack([item['embedding'] for item in batch])
    is_real = torch.stack([item['is_real'] for item in batch])
    
    # Convert content groups to integer IDs for efficient computation
    # Map (source_idx, seg_idx) tuples to unique integers
    content_groups_raw = [item['content_group'] for item in batch]
    unique_groups = sorted(set(content_groups_raw))
    group_to_id = {group: idx for idx, group in enumerate(unique_groups)}
    content_groups = torch.tensor([group_to_id[g] for g in content_groups_raw], dtype=torch.long)
    
    return {
        'embeddings': embeddings,
        'is_real': is_real,
        'content_groups': content_groups,
    }
```

---

### DataLoader Configuration

```python
from torch.utils.data import DataLoader

def create_dataloader(hdf5_path, encoder_name='hubert', batch_size=128, 
                      num_workers=4, shuffle=True):
    """
    Create DataLoader for training.
    
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
        pin_memory=True,
    )
    
    return dataloader
```

**Recommended Batch Size**:
- Since each segment is a sample, you'll have MANY more samples than if averaging
- Use larger batch sizes: 128-256
- This ensures sufficient content group diversity and enough real samples per batch

---

## Training Loop

```python
import torch
from torch.optim import Adam
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device, lambda_var=0.5, lambda_orth=0.1):
    """
    Train for one epoch.
    
    Args:
        model: DisentangledProjector
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        lambda_var: Weight for variance loss
        lambda_orth: Weight for orthogonality loss
    
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
            lambda_orth=lambda_orth
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        for key in epoch_losses:
            epoch_losses[key] += losses_dict[key]
    
    # Average losses
    num_batches = len(dataloader)
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses


def train(model, train_loader, val_loader, num_epochs=50, lr=1e-4, 
          device='cuda', save_dir='./checkpoints'):
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
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
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
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, os.path.join(save_dir, 'best_model.pt'))
            print("Saved best model!")


def validate_epoch(model, dataloader, device, lambda_var=0.5, lambda_orth=0.1):
    """
    Validate for one epoch.
    """
    model.eval()
    
    epoch_losses = {
        'total': 0.0,
        'proto': 0.0,
        'var': 0.0,
        'orth': 0.0,
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            embeddings = batch['embeddings'].to(device)
            is_real = batch['is_real'].to(device)
            content_groups = batch['content_groups'].to(device)
            
            z_auth, z_id = model(embeddings)
            
            total_loss, losses_dict = compute_total_loss(
                z_id, z_auth, is_real, content_groups,
                lambda_var=lambda_var,
                lambda_orth=lambda_orth
            )
            
            for key in epoch_losses:
                epoch_losses[key] += losses_dict[key]
    
    num_batches = len(dataloader)
    for key in epoch_losses:
        epoch_losses[key] /= num_batches
    
    return epoch_losses
```

---

## Main Training Script

```python
# main.py

import torch
from model import DisentangledProjector
from dataset import create_dataloader
from train import train

def main():
    # Configuration
    config = {
        'hdf5_path': '/path/to/your/dataset.h5',
        'encoder_name': 'hubert',  # or 'openl3', 'senet'
        'input_dim': 768,  # Depends on your encoder
        'output_dim': 128,
        'batch_size': 128,
        'num_epochs': 50,
        'lr': 1e-4,
        'lambda_var': 0.5,
        'lambda_orth': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './checkpoints',
    }
    
    # Create dataloaders
    train_loader = create_dataloader(
        config['hdf5_path'],
        encoder_name=config['encoder_name'],
        batch_size=config['batch_size'],
        shuffle=True,
    )
    
    val_loader = create_dataloader(
        config['hdf5_path'].replace('train', 'val'),  # Adjust path as needed
        encoder_name=config['encoder_name'],
        batch_size=config['batch_size'],
        shuffle=False,
    )
    
    # Create model
    model = DisentangledProjector(
        input_dim=config['input_dim'],
        output_dim=config['output_dim']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        lr=config['lr'],
        device=config['device'],
        save_dir=config['save_dir'],
    )

if __name__ == '__main__':
    main()
```

---

## Extracting Trained Embeddings

After training, you'll want to extract z^auth and z^id for all samples:

```python
def extract_embeddings(model, hdf5_path, encoder_name, output_path, device='cuda'):
    """
    Extract z^auth and z^id for all samples and save to new HDF5.
    
    Args:
        model: Trained DisentangledProjector
        hdf5_path: Path to input HDF5
        encoder_name: Encoder used
        output_path: Path to save output HDF5
        device: Device to use
    """
    import h5py
    
    model = model.to(device)
    model.eval()
    
    with h5py.File(hdf5_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        videos_in = f_in['/videos']
        videos_out = f_out.create_group('/videos')
        
        for video_id in tqdm(videos_in.keys(), desc="Extracting embeddings"):
            video_in = videos_in[video_id]
            video_out = videos_out.create_group(video_id)
            
            # Load embeddings
            embeddings = torch.from_numpy(
                video_in['embeddings'][encoder_name][:]
            ).float().to(device)  # [num_augs, num_segs, emb_dim]
            
            num_augs, num_segs, emb_dim = embeddings.shape
            
            # Flatten for batch processing
            embeddings_flat = embeddings.view(-1, emb_dim)  # [num_augs * num_segs, emb_dim]
            
            # Forward pass
            with torch.no_grad():
                z_auth, z_id = model(embeddings_flat)
            
            # Reshape back
            z_auth = z_auth.view(num_augs, num_segs, -1).cpu().numpy()
            z_id = z_id.view(num_augs, num_segs, -1).cpu().numpy()
            
            # Save to output HDF5
            embeddings_group = video_out.create_group('embeddings')
            embeddings_group.create_dataset('z_auth', data=z_auth)
            embeddings_group.create_dataset('z_id', data=z_id)
            
            # Copy metadata and labels
            video_out.copy(video_in['metadata'], 'metadata')
            video_out.copy(video_in['labels'], 'labels')
    
    print(f"Saved disentangled embeddings to {output_path}")
```

---

## Cursor Prompt

Copy this prompt to Cursor to get implementation help:

```markdown
# Task: Implement Disentangled Representation Learning for Deepfake Detection

## Context
I'm implementing disentangled representation learning as described in my thesis (Section 3.3.2). I need to:
1. Train dual projection heads (f_auth and f_id) on pretrained embeddings
2. Use three loss functions: variance minimization, prototypical contrastive, and orthogonality
3. Treat each temporal segment as an independent training sample

## Dataset Structure
- HDF5 files with structure: `/videos/{video_id}/embeddings/{encoder}/`
- Embeddings shape: [num_augmentations, num_segments, embedding_dim]
- Metadata: source_idx (groups augmentations from same source video)
- Labels: audio and video authenticity labels per augmentation
- Each (augmentation, segment) pair is one training sample

## Key Requirements

### 1. Content Grouping
- Content groups: (source_idx, segment_idx)
- Segments at same timestamp across different augmentations have identical content
- Used for prototypical contrastive learning in z^id space

### 2. Variance Minimization
- Collect ALL real segments in batch (regardless of source/timestamp)
- Compute global real centroid
- Minimize variance around this centroid in z^auth space
- Goal: Create one unified "real manifold"

### 3. Architecture
```python
class DisentangledProjector(nn.Module):
    # Two 2-layer MLPs with ReLU
    # Input: embedding_dim (e.g., 768)
    # Output: 128 dimensions
    # f_auth: authenticity projection
    # f_id: identity projection
    # Both outputs should be L2-normalized
```

### 4. Loss Functions

**Variance Loss (Equation 3.6)**:
```python
# For z^auth, real samples only
# L_var = (1/N_real) * sum(||z_i^auth - mu_real||^2)
```

**Prototypical Contrastive Loss (Equation 3.5)**:
```python
# For z^id, all samples
# Group by (source_idx, segment_idx)
# Compute prototypes: c_k = mean(z_i^id for i in group k)
# Use Euclidean distance, temperature=0.1
```

**Orthogonality Loss (Equation 3.3)**:
```python
# Penalize correlation between z^id and z^auth
# L_orth = (1/N^2) * sum_ij |cosine_sim(z_i^id, z_j^auth)|
```

**Total Loss (Equation 3.7)**:
```python
L_total = L_proto + 0.5 * L_var + 0.1 * L_orth
```

### 5. DataLoader
- Batch size: 128-256 (since we have many segments)
- Custom collate function to create content group IDs
- Return: embeddings, is_real mask, content_groups tensor

### 6. Training
- Optimizer: Adam, lr=1e-4
- Train for 50 epochs
- Log all three loss components separately
- Save best model based on validation total loss

## Files to Create

1. `model.py` - DisentangledProjector architecture
2. `losses.py` - Three loss functions + total loss
3. `dataset.py` - HDF5 dataset class + collate function
4. `train.py` - Training and validation loops
5. `main.py` - Main training script
6. `extract_embeddings.py` - Extract z^auth and z^id after training

## Questions to Address

1. How should I handle HDF5 file access efficiently (keep file open vs. open/close per sample)?
2. Should I cache the sample index in memory or rebuild each epoch?
3. How to balance batch composition to ensure sufficient content group diversity?
4. Best way to convert (source_idx, segment_idx) tuples to integer content group IDs?

## Expected Outputs

- Trained model checkpoint with both projection heads
- Training curves for all loss components
- Extracted z^auth and z^id embeddings for entire dataset
- Ability to evaluate disentanglement quality using the 10 metrics from Section 3.4.3

Please help me implement this step-by-step, starting with the dataset class and loss functions.
```

---

## Implementation Checklist

- [ ] Create `model.py` with DisentangledProjector
- [ ] Create `losses.py` with three loss functions
- [ ] Create `dataset.py` with HDF5 dataset class
- [ ] Implement custom collate function
- [ ] Create `train.py` with training loop
- [ ] Create `main.py` to tie everything together
- [ ] Test on small subset of data
- [ ] Train full model
- [ ] Extract disentangled embeddings
- [ ] Evaluate using metrics from Section 3.4.3

---

## Common Issues and Solutions

### Issue 1: Out of Memory
**Solution**: 
- Reduce batch size
- Use gradient accumulation
- Process segments in smaller chunks

### Issue 2: Slow HDF5 Loading
**Solution**:
- Use `pin_memory=True` in DataLoader
- Increase `num_workers`
- Cache frequently accessed data in memory

### Issue 3: Unbalanced Loss Components
**Solution**:
- Adjust λ_var and λ_orth weights
- Monitor individual loss curves
- Ensure sufficient real samples per batch

### Issue 4: Content Groups Too Small
**Solution**:
- Ensure batch size is large enough (128-256)
- Check that multiple augmentations of same source are in batch
- Verify temporal alignment is correct

---

## Next Steps

After training Stage 1 (disentanglement), you'll proceed to Stage 2 (temporal classification):

1. Extract z^auth embeddings for all data
2. Feed z^auth sequences to Transformer encoder (Section 3.3.3)
3. Make frame-level predictions
4. Evaluate on detection performance metrics (Section 3.4.4)

---

## References

- Equation 3.3: Orthogonality constraint
- Equation 3.5: Prototypical contrastive loss
- Equation 3.6: Variance minimization loss
- Equation 3.7: Total loss
- Section 3.3.2: Disentangled representation learning
- Section 3.4.3: Representation quality analysis
- Section 3.4.4: Detection performance

---

**Document Version**: 1.0  
**Last Updated**: November 24, 2025
