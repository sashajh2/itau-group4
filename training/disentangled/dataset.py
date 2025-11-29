"""
HDF5 Dataset for Disentangled Representation Learning.

Each temporal segment is treated as an independent training sample.
Content groups are defined as (source_idx, segment_idx).
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional


class DisentanglementDataset(Dataset):
    """
    Dataset for disentangled representation learning.
    Each temporal segment is treated as an independent sample.
    """
    
    def __init__(self, hdf5_path: str, encoder_name: str = 'hubert', split: Optional[str] = None):
        """
        Args:
            hdf5_path: Path to HDF5 file
            encoder_name: Which encoder embeddings to use ('hubert', 'openl3', 'senet')
            split: Optional split identifier (for future train/val/test splits)
        """
        self.hdf5_path = hdf5_path
        self.encoder_name = encoder_name
        self.split = split
        
        # Build index of all samples
        self.samples: List[Dict] = []
        
        print(f"📂 Loading dataset from {hdf5_path}...")
        print(f"   Encoder: {encoder_name}")
        
        total_videos = 0
        with h5py.File(hdf5_path, 'r') as f:
            if 'videos' not in f:
                raise ValueError(f"HDF5 file must contain 'videos' group")
            
            videos_group = f['/videos']
            video_ids = list(videos_group.keys())
            total_videos = len(video_ids)
            
            for video_idx, video_id in enumerate(video_ids):
                if (video_idx + 1) % 1000 == 0:
                    print(f"   Processed {video_idx + 1}/{total_videos} videos...")
                
                video = videos_group[video_id]
                
                # Load metadata
                if 'augmentation_info' not in video:
                    continue
                    
                source_idx = int(video['augmentation_info'].attrs.get('source_idx', 0))
                
                # Load embeddings
                if f'embeddings/{encoder_name}' not in video:
                    continue
                    
                embeddings = video[f'embeddings/{encoder_name}'][:]  # [num_augs, num_segs, emb_dim]
                
                # Load labels
                audio_labels = video['labels/audio'][:]  # [num_augs, num_segs]
                video_labels = video['labels/video'][:]  # [num_augs, num_segs]
                
                num_augs, num_segs, emb_dim = embeddings.shape
                
                # Verify label shapes match
                if audio_labels.shape != (num_augs, num_segs):
                    print(f"⚠️  Warning: Audio label shape mismatch for {video_id}: "
                          f"expected ({num_augs}, {num_segs}), got {audio_labels.shape}")
                    continue
                
                if video_labels.shape != (num_augs, num_segs):
                    print(f"⚠️  Warning: Video label shape mismatch for {video_id}: "
                          f"expected ({num_augs}, {num_segs}), got {video_labels.shape}")
                    continue
                
                # Create one sample per (augmentation, segment)
                for aug_idx in range(num_augs):
                    for seg_idx in range(num_segs):
                        sample = {
                            'video_id': video_id,
                            'aug_idx': aug_idx,
                            'seg_idx': seg_idx,
                            'source_idx': source_idx,
                            'audio_label': float(audio_labels[aug_idx, seg_idx]),
                            'video_label': float(video_labels[aug_idx, seg_idx]),
                        }
                        self.samples.append(sample)
        
        print(f"✅ Loaded {len(self.samples):,} samples from {total_videos:,} videos")
        
        # Print statistics
        if len(self.samples) > 0:
            real_count = sum(1 for s in self.samples 
                           if s['audio_label'] == 0 and s['video_label'] == 0)
            fake_count = len(self.samples) - real_count
            print(f"   Real samples: {real_count:,} ({100*real_count/len(self.samples):.1f}%)")
            print(f"   Fake samples: {fake_count:,} ({100*fake_count/len(self.samples):.1f}%)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - embedding: torch.Tensor [emb_dim]
                - is_real: torch.Tensor (bool)
                - content_group: Tuple[int, int] (source_idx, seg_idx)
                - source_idx: int
                - seg_idx: int
        """
        sample_info = self.samples[idx]
        
        # Load embedding from HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            video = f['/videos'][sample_info['video_id']]
            embedding = video[f'embeddings/{self.encoder_name}'][
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


def disentanglement_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to create batches for disentangled learning.
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Dictionary with batched tensors:
            - embeddings: torch.Tensor [batch_size, emb_dim]
            - is_real: torch.Tensor [batch_size] (bool)
            - content_groups: torch.Tensor [batch_size] (int, for efficient computation)
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

