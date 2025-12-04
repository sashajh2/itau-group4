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
    
    def __init__(self, hdf5_path: str, encoder_name: str = 'hubert', split: Optional[str] = None,
                 min_samples_per_group: int = 3):
        """
        Args:
            hdf5_path: Path to HDF5 file
            encoder_name: Which encoder embeddings to use ('hubert', 'openl3', 'senet')
            split: Optional split identifier (for future train/val/test splits)
            min_samples_per_group: Minimum number of samples per content group to include
        """
        self.hdf5_path = hdf5_path
        self.encoder_name = encoder_name
        self.split = split
        self.min_samples_per_group = min_samples_per_group
        
        # Build index of all samples
        self.samples: List[Dict] = []
        self.content_group_counts = {}  # Track counts per content group
        
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
                        content_group = (source_idx, seg_idx)
                        sample = {
                            'video_id': video_id,
                            'aug_idx': aug_idx,
                            'seg_idx': seg_idx,
                            'source_idx': source_idx,
                            'audio_label': float(audio_labels[aug_idx, seg_idx]),
                            'video_label': float(video_labels[aug_idx, seg_idx]),
                            'content_group': content_group,
                        }
                        self.samples.append(sample)
                        
                        # Count samples per content group
                        if content_group not in self.content_group_counts:
                            self.content_group_counts[content_group] = 0
                        self.content_group_counts[content_group] += 1
        
        # Filter samples by minimum content group size
        if self.min_samples_per_group > 1:
            initial_count = len(self.samples)
            self.samples = [
                s for s in self.samples
                if self.content_group_counts[s['content_group']] >= self.min_samples_per_group
            ]
            filtered_count = len(self.samples)
            removed = initial_count - filtered_count
            
            print(f"🔍 Filtered to content groups with ≥{self.min_samples_per_group} samples")
            print(f"   Removed {removed:,} samples ({100*removed/initial_count:.1f}%)")
            print(f"   Kept {filtered_count:,} samples ({100*filtered_count/initial_count:.1f}%)")
        
        print(f"✅ Loaded {len(self.samples):,} samples from {total_videos:,} videos")
        
        # Print content group statistics
        if len(self.samples) > 0:
            # Recompute counts for filtered samples
            filtered_group_counts = {}
            for s in self.samples:
                cg = s['content_group']
                filtered_group_counts[cg] = filtered_group_counts.get(cg, 0) + 1
            
            group_sizes = list(filtered_group_counts.values())
            if group_sizes:
                import numpy as np
                print(f"\n📊 Content Group Statistics (after filtering):")
                print(f"   Unique content groups: {len(filtered_group_counts):,}")
                print(f"   Mean samples per group: {np.mean(group_sizes):.2f}")
                print(f"   Median samples per group: {np.median(group_sizes):.2f}")
                print(f"   Min samples per group: {np.min(group_sizes)}")
                print(f"   Max samples per group: {np.max(group_sizes)}")
                print(f"   Groups with ≥2 samples: {sum(1 for s in group_sizes if s >= 2):,}")
                print(f"   Groups with ≥3 samples: {sum(1 for s in group_sizes if s >= 3):,}")
                print(f"   Groups with ≥5 samples: {sum(1 for s in group_sizes if s >= 5):,}")
        
        # Print statistics
        if len(self.samples) > 0:
            real_count = sum(1 for s in self.samples 
                           if s['audio_label'] == 0 and s['video_label'] == 0)
            fake_count = len(self.samples) - real_count
            print(f"\n   Real samples: {real_count:,} ({100*real_count/len(self.samples):.1f}%)")
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
        
        # Content group ID: (source_idx, segment_idx)
        # This ensures segments at same timestamp across augmentations cluster together
        content_group = sample_info['content_group']
        
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

