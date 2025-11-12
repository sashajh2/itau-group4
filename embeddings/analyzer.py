"""
Main analyzer class for deepfake embedding analysis.
Supports both original embeddings and projected embeddings (after model training).
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from typing import Optional, Dict, List
import pandas as pd
from tqdm import tqdm

from embeddings.metrics import (
    analyze_single_timestamp,
    compute_temporal_smoothness,
    label_embedding_correlation
)


class DeepfakeEmbeddingAnalyzer:
    """
    Analyzer for deepfake audio embeddings.
    
    Supports analysis of:
    - Original embeddings (openl3, hubert, senet)
    - Projected embeddings (z_proj, z_context) after model training
    """
    
    def __init__(self, data_path: str, embedding_type: str = 'hubert', 
                 use_projection: Optional[str] = None):
        """
        Args:
            data_path: Path to HDF5 file (exports/deepfake_embeddings.h5)
            embedding_type: 'openl3', 'hubert', or 'senet' (for original embeddings)
            use_projection: None (original), 'z_proj', or 'z_context' (for projected)
        """
        self.data_path = data_path
        self.embedding_type = embedding_type
        self.use_projection = use_projection
        self.data = self._load_metadata()
        self.pca_models = {}
        self.umap_models = {}
    
    def _load_metadata(self) -> Dict:
        """Load metadata from HDF5 file"""
        metadata = {}
        with h5py.File(self.data_path, 'r') as f:
            # Load global metadata
            meta_grp = f['metadata']
            metadata['video_ids'] = [vid.decode() if isinstance(vid, bytes) else vid 
                                    for vid in meta_grp['video_ids'][:]]
            metadata['total_videos'] = meta_grp.attrs['total_videos']
            
        return metadata
    
    def get_video_data(self, video_id: str) -> Dict:
        """Load a specific video's data from HDF5"""
        safe_id = video_id.replace('/', '_')
        
        with h5py.File(self.data_path, 'r') as f:
            if safe_id not in f['videos']:
                raise ValueError(f"Video {video_id} not found in HDF5 file")
            
            vid_grp = f['videos'][safe_id]
            
            # Load metadata
            data = {
                'dataset': vid_grp.attrs['dataset'].decode() if isinstance(vid_grp.attrs['dataset'], bytes) else vid_grp.attrs['dataset'],
                'num_segments': int(vid_grp.attrs['num_segments']),
                'num_augmentations': int(vid_grp.attrs['num_augmentations']),
                'augmentation_info': {
                    'video_paths': [p.decode() if isinstance(p, bytes) else p 
                                   for p in vid_grp['augmentation_info/video_paths'][:]],
                    'types': [t.decode() if isinstance(t, bytes) else t 
                             for t in vid_grp['augmentation_info/types'][:]],
                    'source_idx': int(vid_grp['augmentation_info'].attrs['source_idx'])
                },
                'labels': {
                    'audio': vid_grp['labels/audio'][:],
                    'video': vid_grp['labels/video'][:]
                }
            }
            
            # Load embeddings based on configuration
            if self.use_projection is None:
                # Original embeddings
                data['embeddings'] = vid_grp[f'embeddings/{self.embedding_type}'][:]
            else:
                # Projected embeddings (check if exists)
                if f'embeddings/{self.use_projection}' in vid_grp:
                    data['embeddings'] = vid_grp[f'embeddings/{self.use_projection}'][:]
                else:
                    raise ValueError(f"Projection {self.use_projection} not found for video {video_id}")
            
            return data
    
    def get_embeddings(self, video_id: str) -> np.ndarray:
        """Get embeddings for a video [num_augs, num_segs, emb_dim]"""
        return self.get_video_data(video_id)['embeddings']
    
    def fit_global_pca(self, n_components: int = 50, sample_size: int = 50000):
        """
        Fit PCA on a sample of all embeddings.
        
        Args:
            n_components: Number of PCA components
            sample_size: Number of embedding vectors to sample
        
        Returns:
            Fitted PCA model
        """
        print(f"Fitting PCA (n_components={n_components})...")
        
        # Collect embeddings
        all_embeddings = []
        total_collected = 0
        
        with h5py.File(self.data_path, 'r') as f:
            video_ids = [vid.decode() if isinstance(vid, bytes) else vid 
                        for vid in f['metadata']['video_ids'][:]]
            
            for video_id in tqdm(video_ids, desc="Collecting embeddings"):
                if total_collected >= sample_size:
                    break
                
                safe_id = video_id.replace('/', '_')
                
                try:
                    if self.use_projection is None:
                        emb = f[f'videos/{safe_id}/embeddings/{self.embedding_type}'][:]
                    else:
                        if f'videos/{safe_id}/embeddings/{self.use_projection}' not in f:
                            continue
                        emb = f[f'videos/{safe_id}/embeddings/{self.use_projection}'][:]
                    
                    # Flatten: [num_augs, num_segs, emb_dim] -> [num_augs*num_segs, emb_dim]
                    emb_flat = emb.reshape(-1, emb.shape[-1])
                    
                    # Sample if needed
                    if total_collected + len(emb_flat) > sample_size:
                        remaining = sample_size - total_collected
                        idx = np.random.choice(len(emb_flat), remaining, replace=False)
                        emb_flat = emb_flat[idx]
                    
                    all_embeddings.append(emb_flat)
                    total_collected += len(emb_flat)
                except KeyError:
                    # Skip videos without this embedding type
                    continue
        
        if len(all_embeddings) == 0:
            raise ValueError("No embeddings found to fit PCA")
        
        # Stack and fit PCA
        all_embeddings = np.vstack(all_embeddings)
        print(f"Fitting PCA on {len(all_embeddings)} vectors...")
        
        pca = PCA(n_components=n_components)
        pca.fit(all_embeddings)
        
        key = self.use_projection if self.use_projection else self.embedding_type
        self.pca_models[key] = pca
        
        print(f"✓ Explained variance (first 2 PCs): {pca.explained_variance_ratio_[:2].sum():.3f}")
        return pca
    
    def experiment1_cross_aug_single_timestamp(self, video_id: str, timestamp_idx: int,
                                               save_fig: Optional[str] = None) -> Optional[Dict]:
        """
        Experiment 1: Analyze embeddings across all augmentations at a single timestamp.
        
        Only works for videos with multiple augmentations (AVDeepfake1M).
        ShareVeo3 videos (1 augmentation) are skipped.
        
        Args:
            video_id: Video identifier
            timestamp_idx: Segment index (0 to num_segments-1)
            save_fig: Optional path to save figure
        
        Returns:
            Dictionary with computed metrics, or None if skipped
        """
        video_data = self.get_video_data(video_id)
        
        # Check if video has multiple augmentations
        if video_data['num_augmentations'] == 1:
            print(f"⚠️  Skipping {video_id}: Only 1 augmentation (ShareVeo3 video)")
            return None
        
        # Check timestamp index
        if timestamp_idx >= video_data['num_segments']:
            raise ValueError(f"timestamp_idx {timestamp_idx} >= num_segments {video_data['num_segments']}")
        
        # Extract embeddings at this timestamp: [num_augs, emb_dim]
        embeddings = video_data['embeddings'][:, timestamp_idx, :]
        labels = video_data['labels']['audio'][:, timestamp_idx]
        source_idx = video_data['augmentation_info']['source_idx']
        
        # Compute metrics
        metrics = analyze_single_timestamp(embeddings, labels, source_idx)
        
        # Transform to PCA space for visualization
        key = self.use_projection if self.use_projection else self.embedding_type
        if key in self.pca_models:
            embeddings_pca = self.pca_models[key].transform(embeddings)
            
            from embeddings.visualization import plot_cross_augmentation_timestamp
            # Estimate segment duration (default 0.15s, can be improved)
            segment_duration = 0.15
            fig = plot_cross_augmentation_timestamp(
                embeddings_pca, labels, source_idx,
                video_id, timestamp_idx * segment_duration,
                self.embedding_type
            )
            if save_fig:
                fig.savefig(save_fig, dpi=300, bbox_inches='tight')
                plt.close(fig)
        else:
            print("Warning: PCA not fitted, skipping visualization")
        
        metrics['video_id'] = video_id
        metrics['timestamp_idx'] = timestamp_idx
        
        return metrics
    
    def experiment2_single_video_temporal(self, video_id: str, augmentation_idx: int,
                                         save_fig: Optional[str] = None) -> Dict:
        """
        Experiment 2: Analyze single video over time (temporal evolution).
        
        Works for both AVDeepfake1M and ShareVeo3.
        
        Args:
            video_id: Video identifier
            augmentation_idx: Which augmentation to analyze (0 for ShareVeo3)
            save_fig: Optional path to save figure
        
        Returns:
            Dictionary with temporal metrics
        """
        video_data = self.get_video_data(video_id)
        
        # Check augmentation index
        if augmentation_idx >= video_data['num_augmentations']:
            raise ValueError(f"augmentation_idx {augmentation_idx} >= num_augmentations {video_data['num_augmentations']}")
        
        # Extract sequence: [num_segs, emb_dim]
        embeddings_seq = video_data['embeddings'][augmentation_idx]
        labels_seq = video_data['labels']['audio'][augmentation_idx]
        video_path = video_data['augmentation_info']['video_paths'][augmentation_idx]
        
        # Compute metrics
        metrics = {
            'video_id': video_id,
            'augmentation_idx': augmentation_idx,
            'video_path': video_path,
            'temporal_smoothness': compute_temporal_smoothness(embeddings_seq),
            'label_correlation': label_embedding_correlation(embeddings_seq, labels_seq)
        }
        
        # Transform to PCA for visualization
        key = self.use_projection if self.use_projection else self.embedding_type
        if key in self.pca_models:
            embeddings_pca = self.pca_models[key].transform(embeddings_seq)
            
            from embeddings.visualization import plot_single_video_temporal
            fig = plot_single_video_temporal(embeddings_pca, labels_seq, video_path)
            if save_fig:
                fig.savefig(save_fig, dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        return metrics
    
    def experiment3_aggregate_analysis(self, max_videos: Optional[int] = None) -> pd.DataFrame:
        """
        Experiment 3: Compute aggregate statistics across all videos and timestamps.
        
        Args:
            max_videos: Optional limit on number of videos to analyze
        
        Returns:
            DataFrame with metrics per (video_id, timestamp_idx)
        """
        results = []
        
        video_ids = self.data['video_ids']
        if max_videos:
            video_ids = video_ids[:max_videos]
        
        for video_id in tqdm(video_ids, desc="Analyzing videos"):
            try:
                video_data = self.get_video_data(video_id)
                
                # Skip single-augmentation videos for cross-aug analysis
                if video_data['num_augmentations'] == 1:
                    continue
                
                num_segments = video_data['num_segments']
                
                for timestamp_idx in range(num_segments):
                    embeddings = video_data['embeddings'][:, timestamp_idx, :]
                    labels = video_data['labels']['audio'][:, timestamp_idx]
                    source_idx = video_data['augmentation_info']['source_idx']
                    
                    # Skip if all same label (no variation)
                    if len(np.unique(labels)) < 2:
                        continue
                    
                    metrics = analyze_single_timestamp(embeddings, labels, source_idx)
                    metrics['video_id'] = video_id
                    metrics['timestamp_idx'] = timestamp_idx
                    metrics['timestamp_seconds'] = timestamp_idx * 0.15  # Default segment duration
                    
                    results.append(metrics)
            except (KeyError, ValueError) as e:
                # Skip videos with missing data
                print(f"⚠️  Skipping {video_id}: {e}")
                continue
        
        return pd.DataFrame(results)

