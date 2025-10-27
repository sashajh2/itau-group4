"""
Data loading and preprocessing for deepfake detection experiments.

This module provides functions to:
1. Load embeddings from Neon Postgres
2. Create PyTorch datasets and dataloaders
3. Split data into train/val/test sets
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from retriever.retriever import load_embedding_data


class KWayNShotBatchSampler(Sampler):
    """
    Batch sampler for episodic training (K-way N-shot learning).
    
    Each batch/episode contains:
    - N distinct identities (ways)
    - K support samples per identity
    - Q query samples per identity
    
    Total samples per episode: N * (K + Q)
    """
    
    def __init__(self, dataset, n_way, k_shot, n_query, episodes_per_epoch):
        """
        Args:
            dataset: EmbeddingDataset instance
            n_way: Number of identities per episode
            k_shot: Number of support samples per identity
            n_query: Number of query samples per identity
            episodes_per_epoch: Number of episodes per epoch
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.episodes_per_epoch = episodes_per_epoch
        
        # Filter out identities with too few samples
        self.valid_ids = [
            id_idx for id_idx in dataset.unique_ids
            if len(dataset.id_to_indices[id_idx]) >= (k_shot + n_query)
        ]
        
        if len(self.valid_ids) < n_way:
            raise ValueError(
                f"Not enough valid identities ({len(self.valid_ids)}) "
                f"for {n_way}-way sampling. Need at least {n_way} identities "
                f"with {k_shot + n_query} samples each."
            )
    
    def __iter__(self):
        """Generate episodes"""
        for _ in range(self.episodes_per_epoch):
            batch_indices = []
            
            # Sample N identities
            sampled_ids = np.random.choice(self.valid_ids, self.n_way, replace=False)
            
            # For each identity, sample K support + Q query examples
            for identity_idx in sampled_ids:
                available_indices = self.dataset.id_to_indices[identity_idx]
                
                selected_indices = np.random.choice(
                    available_indices,
                    self.k_shot + self.n_query,
                    replace=False
                )
                batch_indices.extend(selected_indices)
            
            yield batch_indices
    
    def __len__(self):
        return self.episodes_per_epoch


def collate_simple(batch):
    """Collate function for DataLoader"""
    e = torch.stack([b["e"] for b in batch], dim=0)
    y = torch.tensor([float(b["y"]) for b in batch], dtype=torch.float32)
    id_idx = torch.tensor([int(b["id_idx"]) for b in batch], dtype=torch.int64)
    is_real = torch.tensor([int(b["is_real"]) for b in batch], dtype=torch.int64)
    
    identity = [b["identity"] for b in batch]
    segment_id = [b["segment_id"] for b in batch]
    
    return {
        "e": e, 
        "y": y, 
        "id_idx": id_idx, 
        "is_real": is_real,
        "identity": identity, 
        "segment_id": segment_id
    }


class EmbeddingDataset(Dataset):
    """PyTorch Dataset for embeddings"""
    
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        
        # Pre-extract arrays to avoid .iloc overhead
        self.emb = [np.asarray(x, dtype=np.float32) for x in self.df['embedding'].tolist()]
        self.y = self.df['label'].astype(np.float32).to_numpy()
        self.idi = self.df['id_idx'].astype(np.int64).to_numpy()
        self.real = (self.df['label'] == 1).astype(np.int64).to_numpy()
        self.ident = self.df['identity'].tolist()
        self.seg_ids = self.df['segment_id'].tolist()
        
        self.d = int(self.emb[0].shape[0]) if self.emb else None
        
        # Build id_to_indices mapping for episodic sampling
        # This maps each identity index to all sample indices that belong to that identity
        # Used by KWayNShotBatchSampler to efficiently sample episodes
        self.id_to_indices = defaultdict(list)
        for i, id_idx in enumerate(self.idi):
            self.id_to_indices[id_idx].append(i)
        self.unique_ids = list(self.id_to_indices.keys())
        
        print(f"Dataset: {len(self)} samples, {len(self.unique_ids)} unique identities")
        print(f"Identity distribution: {[len(indices) for indices in list(self.id_to_indices.values())[:5]]}...")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return {
            "e": torch.from_numpy(self.emb[i]),
            "y": torch.tensor(self.y[i]),
            "id_idx": torch.tensor(self.idi[i]),
            "is_real": torch.tensor(self.real[i]),
            "identity": self.ident[i],
            "segment_id": self.seg_ids[i],
        }


def balanced_copy(x: pd.DataFrame) -> pd.DataFrame:
    """Balance dataset by downsampling majority class"""
    pos = x[x['label'] == 1]
    neg = x[x['label'] == 0]
    
    if len(pos) == 0 or len(neg) == 0:
        return x.reset_index(drop=True)
    
    if len(pos) > len(neg):
        pos = pos.sample(n=len(neg), random_state=123)
    else:
        neg = neg.sample(n=len(pos), random_state=123)
    
    out = pd.concat([pos, neg]).sample(frac=1.0, random_state=123).reset_index(drop=True)
    return out


def load_data(model_name: str = "openl3", version: str = "2025-09-12", 
              noise: str = "none", denoiser_name: str = "none",
              training_config: Dict[str, Any] = None,
              evaluation_config: Dict[str, Any] = None,
              num_workers: int = 4) -> Dict[str, DataLoader]:
    """
    Load embeddings from Neon and create PyTorch DataLoaders for training and evaluation.
    
    Args:
        model_name: Model name ('hubert', 'openl3', 'senet')
        version: Version string
        noise: Noise level filter
        denoiser_name: Denoiser filter
        training_config: Training configuration dict with batching parameters
        evaluation_config: Evaluation configuration dict with parameters
        num_workers: Number of worker processes
    
    Returns:
        Dictionary of DataLoaders with keys:
        - 'train': Standard shuffled training DataLoader (used for most stages)
        - 'train_het_stage_b': Episodic training for heterogeneous head Stage B (prototypical)
        - 'val': Standard validation DataLoader
        - 'val_het_eval': Episodic evaluation for heterogeneous head
        - 'test': Standard test DataLoader
        - 'test_het_eval': Episodic evaluation for heterogeneous head
    """
    
    print("Loading embeddings from Neon...")
    embeddings, labels, video_ids, segment_ids = load_embedding_data(
        model_name=model_name,
        version=version,
        noise=noise if noise != "none" else None,
        denoiser_name=denoiser_name if denoiser_name != "none" else None,
    )
    
    print(f"Loaded {len(segment_ids)} embeddings with dimension {embeddings.shape[1]}")
    
    # video_id is already the identity
    # Create DataFrame
    df = pd.DataFrame({
        'embedding': [row for row in embeddings],
        'label': labels,
        'identity': video_ids,  # video_id IS the identity
        'segment_id': segment_ids
    })
    
    print(f"Total rows: {len(df)}")
    print(f"Label counts (1=real, 0=fake):")
    print(df['label'].value_counts(dropna=False).to_string())
    print(f"Unique identities (video_ids): {df['identity'].nunique()}")
    
    # Map identity -> int
    id2idx = {s: i for i, s in enumerate(sorted(df['identity'].unique()))}
    df['id_idx'] = df['identity'].map(id2idx).astype(np.int64)
    
    # Split data by identity first (ensure each set has enough samples per identity)
    # Filter identities that have at least min_samples_per_id samples
    min_samples_per_id = 20  # Minimum for episodic evaluation
    identity_counts = df.groupby('identity').size()
    valid_identities = identity_counts[identity_counts >= min_samples_per_id].index
    
    # Filter dataframe to only include identities with enough samples
    df_filtered = df[df['identity'].isin(valid_identities)].copy()
    
    print(f"\nAfter filtering for identities with >= {min_samples_per_id} samples:")
    print(f"  Total samples: {len(df_filtered)} (from {len(df)} original)")
    print(f"  Unique identities: {df_filtered['identity'].nunique()} (from {df['identity'].nunique()} original)")
    print(f"  Label counts: {df_filtered['label'].value_counts().to_dict()}")
    
    # Split identities (not samples) to ensure each split has enough identities
    unique_ids_list = df_filtered['identity'].unique()
    n_ids = len(unique_ids_list)
    n_train_ids = int(0.70 * n_ids)
    n_val_ids = int(0.15 * n_ids)
    
    # Shuffle and split identities
    import random
    random.seed(123)
    random.shuffle(unique_ids_list)
    
    train_ids = set(unique_ids_list[:n_train_ids])
    val_ids = set(unique_ids_list[n_train_ids:n_train_ids + n_val_ids])
    test_ids = set(unique_ids_list[n_train_ids + n_val_ids:])
    
    # Split dataframe by identities
    train_df = df_filtered[df_filtered['identity'].isin(train_ids)].copy()
    val_df = df_filtered[df_filtered['identity'].isin(val_ids)].copy()
    test_df = df_filtered[df_filtered['identity'].isin(test_ids)].copy()
    
    # Check that each set has enough samples per identity
    print("\n=== Data Splits ===")
    for name, x in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name:>5}: {len(x):7,} rows | ids={x['identity'].nunique():5,}")
        print("       label counts:", x['label'].value_counts().to_dict())
        
        # Check minimum samples per identity in this split
        min_samples = x.groupby('identity').size().min()
        print(f"       min samples per identity: {min_samples}")
        
        # Count identities with at least 20 samples
        id_counts = x.groupby('identity').size()
        ids_with_20_plus = (id_counts >= 20).sum()
        print(f"       identities with >= 20 samples: {ids_with_20_plus}/{len(id_counts)}")
    
    # Create datasets
    train_dataset = EmbeddingDataset(train_df)
    val_dataset = EmbeddingDataset(val_df)
    test_dataset = EmbeddingDataset(test_df)
    
    # Get config parameters
    if training_config is None:
        training_config = {}
    if evaluation_config is None:
        evaluation_config = {}
    
    # Default parameters
    batch_size = training_config.get('batch_size', 256)
    
    # Episodic training parameters (for heterogeneous head Stage B only)
    het_n_way = training_config.get('het_n_way', 5)
    het_k_shot = training_config.get('het_k_shot', 5)
    het_n_query = training_config.get('het_n_query', 15)
    het_episodes_per_epoch = training_config.get('het_episodes_per_epoch', 100)
    
    # Evaluation episodic parameters
    eval_n_way = evaluation_config.get('eval_n_way', 5)
    eval_k_shot = evaluation_config.get('eval_k_shot', 5)
    eval_n_query = evaluation_config.get('eval_n_query', 15)
    eval_episodes = evaluation_config.get('eval_episodes', 50)
    
    pin_memory = torch.cuda.is_available()
    
    # ========== TRAINING LOADERS ==========
    
    # Standard training DataLoader (used for all Stage A and Stage B except het Stage B)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_simple,
        pin_memory=pin_memory
    )
    
    # Episodic training for heterogeneous head Stage B (prototypical)
    train_sampler_het_stage_b = KWayNShotBatchSampler(
        train_dataset, het_n_way, het_k_shot, het_n_query, het_episodes_per_epoch
    )
    train_het_stage_b = DataLoader(
        train_dataset,
        batch_sampler=train_sampler_het_stage_b,
        num_workers=num_workers,
        collate_fn=collate_simple,
        pin_memory=pin_memory
    )
    
    # ========== VALIDATION LOADERS ==========
    
    # Standard validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_simple,
        pin_memory=pin_memory
    )
    
    # Episodic evaluation for heterogeneous head
    val_sampler_het_eval = KWayNShotBatchSampler(
        val_dataset, eval_n_way, eval_k_shot, eval_n_query, eval_episodes
    )
    val_het_eval = DataLoader(
        val_dataset,
        batch_sampler=val_sampler_het_eval,
        num_workers=num_workers,
        collate_fn=collate_simple,
        pin_memory=pin_memory
    )
    
    # ========== TEST LOADERS ==========
    
    # Standard test
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_simple,
        pin_memory=pin_memory
    )
    
    # Episodic evaluation for heterogeneous head
    test_sampler_het_eval = KWayNShotBatchSampler(
        test_dataset, eval_n_way, eval_k_shot, eval_n_query, eval_episodes
    )
    test_het_eval = DataLoader(
        test_dataset,
        batch_sampler=test_sampler_het_eval,
        num_workers=num_workers,
        collate_fn=collate_simple,
        pin_memory=pin_memory
    )
    
    return {
        'train': train_loader,
        'train_het_stage_b': train_het_stage_b,
        'val': val_loader,
        'val_het_eval': val_het_eval,
        'test': test_loader,
        'test_het_eval': test_het_eval,
    }

