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
from torch.utils.data import Dataset, DataLoader

from retriever.retriever import load_embedding_data


def collate_simple(batch):
    """Collate function for DataLoader"""
    e = torch.stack([b["e"] for b in batch], dim=0)
    y = torch.tensor([float(b["y"]) for b in batch], dtype=torch.float32)
    id_idx = torch.tensor([int(b["id_idx"]) for b in batch], dtype=torch.int64)
    is_real = torch.tensor([int(b["is_real"]) for b in batch], dtype=torch.int64)
    
    identity = [b["identity"] for b in batch]
    video_id = [b["video_id"] for b in batch]
    
    return {
        "e": e, 
        "y": y, 
        "id_idx": id_idx, 
        "is_real": is_real,
        "identity": identity, 
        "video_id": video_id
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
              batch_size: int = 256, num_workers: int = 4,
              val_test_balanced: bool = True) -> Dict[str, DataLoader]:
    """
    Load embeddings from Neon and create PyTorch DataLoaders.
    
    Args:
        model_name: Model name ('hubert', 'openl3', 'senet')
        version: Version string
        noise: Noise level filter
        denoiser_name: Denoiser filter
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        val_test_balanced: Whether to balance validation and test sets
    
    Returns:
        Dictionary of DataLoaders: {'train': train_loader, 'val': val_loader, 'test': test_loader}
    """
    
    print("Loading embeddings from Neon...")
    embeddings, labels, video_ids, segment_ids = load_embedding_data(
        model_name=model_name,
        version=version,
        noise=noise if noise != "none" else None,
        denoiser_name=denoiser_name if denoiser_name != "none" else None,
    )
    
    print(f"Loaded {len(video_ids)} embeddings with dimension {embeddings.shape[1]}")
    
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
    
    # Split data (70/15/15)
    n = len(df)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    
    train_df, hold_df = train_test_split(
        df, test_size=(n - n_train), random_state=123, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        hold_df, test_size=(len(hold_df) - n_val), random_state=123, stratify=hold_df['label']
    )
    
    print("\n=== Data Splits ===")
    for name, x in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name:>5}: {len(x):7,} rows | ids={x['identity'].nunique():5,}")
        print("       label counts:", x['label'].value_counts().to_dict())
    
    # Optionally balance val/test
    if val_test_balanced:
        val_df = balanced_copy(val_df)
        test_df = balanced_copy(test_df)
        
        print("\n=== After Balancing val/test ===")
        for name, x in [("val", val_df), ("test", test_df)]:
            print(f"{name:>8}: {len(x):7,} rows | ids={x['identity'].nunique():5,}")
            print("          label counts:", x['label'].value_counts().to_dict())
    
    # Create datasets
    train_dataset = EmbeddingDataset(train_df)
    val_dataset = EmbeddingDataset(val_df)
    test_dataset = EmbeddingDataset(test_df)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_simple,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_simple,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_simple,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

