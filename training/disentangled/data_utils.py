"""
Shared data loading utilities for disentangled representation learning.
"""
import h5py
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional


def load_data_from_hdf5(
    hdf5_path: str,
    encoder_name: str = 'hubert',
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings, labels, and content groups from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        encoder_name: Encoder name ('hubert', 'openl3', 'senet')
        max_samples: Maximum number of samples to load (None = all)
    
    Returns:
        embeddings: np.ndarray [n_samples, emb_dim]
        labels: np.ndarray [n_samples] (0=real, 1=fake)
        content_groups: np.ndarray [n_samples] (content group IDs)
    """
    print(f"📂 Loading data from {hdf5_path}...")
    print(f"   Encoder: {encoder_name}")
    
    all_embeddings = []
    all_labels = []
    all_content_groups = []
    content_group_map = {}  # Map (source_idx, seg_idx) to integer ID
    next_group_id = 0
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'videos' not in f:
            raise ValueError(f"HDF5 file must contain 'videos' group")
        
        videos_group = f['/videos']
        video_ids = list(videos_group.keys())
        total_videos = len(video_ids)
        
        print(f"   Found {total_videos:,} videos")
        
        for video_idx, video_id in enumerate(tqdm(video_ids, desc="Loading videos")):
            if max_samples and len(all_embeddings) >= max_samples:
                break
            
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
            
            # Create one sample per (augmentation, segment)
            for aug_idx in range(num_augs):
                if max_samples and len(all_embeddings) >= max_samples:
                    break
                
                for seg_idx in range(num_segs):
                    if max_samples and len(all_embeddings) >= max_samples:
                        break
                    
                    # Content group: (source_idx, seg_idx)
                    content_group_key = (source_idx, seg_idx)
                    if content_group_key not in content_group_map:
                        content_group_map[content_group_key] = next_group_id
                        next_group_id += 1
                    
                    content_group_id = content_group_map[content_group_key]
                    
                    # Embedding
                    embedding = embeddings[aug_idx, seg_idx]
                    all_embeddings.append(embedding)
                    
                    # Label: is_real = (audio_label == 0 and video_label == 0)
                    is_real = ((audio_labels[aug_idx, seg_idx] == 0) and 
                               (video_labels[aug_idx, seg_idx] == 0))
                    label = 0 if is_real else 1  # 0=real, 1=fake
                    all_labels.append(label)
                    
                    # Content group
                    all_content_groups.append(content_group_id)
    
    embeddings = np.array(all_embeddings)
    labels = np.array(all_labels, dtype=int)
    content_groups = np.array(all_content_groups, dtype=int)
    
    print(f"✅ Loaded {len(embeddings):,} samples")
    print(f"   Real samples: {np.sum(labels == 0):,} ({100*np.sum(labels == 0)/len(labels):.1f}%)")
    print(f"   Fake samples: {np.sum(labels == 1):,} ({100*np.sum(labels == 1)/len(labels):.1f}%)")
    print(f"   Unique content groups: {len(np.unique(content_groups)):,}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings, labels, content_groups

