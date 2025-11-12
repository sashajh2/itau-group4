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
        Fit PCA on a balanced sample of all embeddings (real and fake).
        
        Args:
            n_components: Number of PCA components
            sample_size: Number of embedding vectors to sample (will be split between real/fake)
        
        Returns:
            Fitted PCA model
        """
        print(f"Fitting PCA (n_components={n_components}) on balanced real/fake sample...")
        
        # Collect embeddings with labels
        real_embeddings = []
        fake_embeddings = []
        
        with h5py.File(self.data_path, 'r') as f:
            video_ids = [vid.decode() if isinstance(vid, bytes) else vid 
                        for vid in f['metadata']['video_ids'][:]]
            
            for video_id in tqdm(video_ids, desc="Collecting embeddings"):
                safe_id = video_id.replace('/', '_')
                
                try:
                    if self.use_projection is None:
                        emb = f[f'videos/{safe_id}/embeddings/{self.embedding_type}'][:]
                    else:
                        if f'videos/{safe_id}/embeddings/{self.use_projection}' not in f:
                            continue
                        emb = f[f'videos/{safe_id}/embeddings/{self.use_projection}'][:]
                    
                    labels = f[f'videos/{safe_id}/labels/audio'][:]
                    
                    # Flatten: [num_augs, num_segs, emb_dim] -> [num_augs*num_segs, emb_dim]
                    emb_flat = emb.reshape(-1, emb.shape[-1])
                    labels_flat = labels.flatten()
                    
                    # Separate real and fake
                    is_real = labels_flat == 0
                    is_fake = labels_flat > 0
                    
                    if is_real.any():
                        real_embeddings.append(emb_flat[is_real])
                    if is_fake.any():
                        fake_embeddings.append(emb_flat[is_fake])
                except KeyError:
                    # Skip videos without this embedding type
                    continue
        
        # Get embedding dimension from first available embedding
        if len(real_embeddings) > 0:
            emb_dim = real_embeddings[0].shape[-1]
        elif len(fake_embeddings) > 0:
            emb_dim = fake_embeddings[0].shape[-1]
        else:
            raise ValueError("No embeddings found to fit PCA")
        
        # Stack real and fake
        if len(real_embeddings) > 0:
            real_embeddings = np.vstack(real_embeddings)
        else:
            real_embeddings = np.array([]).reshape(0, emb_dim)
        
        if len(fake_embeddings) > 0:
            fake_embeddings = np.vstack(fake_embeddings)
        else:
            fake_embeddings = np.array([]).reshape(0, emb_dim)
        
        print(f"   Collected {len(real_embeddings):,} real and {len(fake_embeddings):,} fake embeddings")
        
        # Sample balanced amounts
        samples_per_class = sample_size // 2
        if len(real_embeddings) > 0:
            if len(real_embeddings) > samples_per_class:
                real_idx = np.random.choice(len(real_embeddings), samples_per_class, replace=False)
                real_sample = real_embeddings[real_idx]
            else:
                real_sample = real_embeddings
        else:
            real_sample = np.array([]).reshape(0, emb_dim)
        
        if len(fake_embeddings) > 0:
            if len(fake_embeddings) > samples_per_class:
                fake_idx = np.random.choice(len(fake_embeddings), samples_per_class, replace=False)
                fake_sample = fake_embeddings[fake_idx]
            else:
                fake_sample = fake_embeddings
        else:
            fake_sample = np.array([]).reshape(0, emb_dim)
        
        # Combine and fit PCA
        if len(real_sample) > 0 and len(fake_sample) > 0:
            all_embeddings = np.vstack([real_sample, fake_sample])
        elif len(real_sample) > 0:
            all_embeddings = real_sample
        elif len(fake_sample) > 0:
            all_embeddings = fake_sample
        else:
            raise ValueError("No embeddings found to fit PCA")
        
        print(f"Fitting PCA on {len(all_embeddings):,} vectors ({len(real_sample):,} real, {len(fake_sample):,} fake)...")
        
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
    
    def compute_aggregate_statistics(self, max_videos: Optional[int] = None) -> Dict:
        """
        Compute aggregate statistics across all AVDeepfake1M videos.
        
        Computes:
        1. Average cosine similarity of real augmentations to source vs fake augmentations to source
        2. Overall linear separability (label 0 vs label != 0)
        
        Args:
            max_videos: Optional limit on number of videos to analyze
        
        Returns:
            Dictionary with aggregate statistics
        """
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        real_to_source_similarities = []
        fake_to_source_similarities = []
        all_embeddings = []
        all_binary_labels = []
        videos_analyzed = 0
        
        video_ids = self.data['video_ids']
        if max_videos:
            video_ids = video_ids[:max_videos]
        
        print("Computing aggregate statistics...")
        for video_id in tqdm(video_ids, desc="Processing videos"):
            try:
                video_data = self.get_video_data(video_id)
                
                # Skip single-augmentation videos (ShareVeo3)
                if video_data['num_augmentations'] == 1:
                    continue
                
                # Skip if not AVDeepfake1M
                if video_data['dataset'] != 'avdeepfake1m':
                    continue
                
                videos_analyzed += 1
                num_segments = video_data['num_segments']
                source_idx = video_data['augmentation_info']['source_idx']
                
                for timestamp_idx in range(num_segments):
                    embeddings = video_data['embeddings'][:, timestamp_idx, :]  # [num_augs, emb_dim]
                    labels = video_data['labels']['audio'][:, timestamp_idx]  # [num_augs]
                    
                    # Skip if all same label (no variation)
                    if len(np.unique(labels)) < 2:
                        continue
                    
                    # Get source embedding
                    source_embedding = embeddings[source_idx]
                    
                    # Compute cosine similarity to source
                    cos_sim = cosine_similarity(embeddings, source_embedding.reshape(1, -1)).flatten()
                    
                    # Separate real and fake
                    is_real = labels == 0
                    is_fake = labels > 0
                    
                    if is_real.any():
                        real_to_source_similarities.extend(cos_sim[is_real].tolist())
                    if is_fake.any():
                        fake_to_source_similarities.extend(cos_sim[is_fake].tolist())
                    
                    # Collect for linear separability analysis
                    all_embeddings.append(embeddings)
                    all_binary_labels.append((labels > 0).astype(int))
                    
            except (KeyError, ValueError) as e:
                continue
        
        # Flatten embeddings and labels for linear separability
        if all_embeddings:
            all_embeddings_flat = np.vstack(all_embeddings)  # [total_augs, emb_dim]
            all_binary_labels_flat = np.concatenate(all_binary_labels)  # [total_augs]
            
            # Compute linear separability
            if len(np.unique(all_binary_labels_flat)) > 1:
                # Check class balance to determine appropriate CV
                unique_labels, counts = np.unique(all_binary_labels_flat, return_counts=True)
                min_class_count = counts.min()
                
                # Use stratified CV, but limit folds based on smallest class
                max_cv = min(5, min_class_count)  # Can't have more folds than samples in smallest class
                if max_cv < 2:
                    # Not enough samples for CV, use simple train/test split
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        all_embeddings_flat, all_binary_labels_flat, 
                        test_size=0.2, random_state=42, stratify=all_binary_labels_flat
                    )
                    clf = LogisticRegression(max_iter=1000, random_state=42)
                    clf.fit(X_train, y_train)
                    linear_separability = float(clf.score(X_test, y_test))
                else:
                    from sklearn.model_selection import StratifiedKFold
                    clf = LogisticRegression(max_iter=1000, random_state=42)
                    cv = StratifiedKFold(n_splits=max_cv, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(clf, all_embeddings_flat, all_binary_labels_flat, cv=cv)
                    linear_separability = float(cv_scores.mean())
            else:
                linear_separability = None
        else:
            linear_separability = None
        
        # Compute statistics
        stats = {
            'num_real_samples': len(real_to_source_similarities),
            'num_fake_samples': len(fake_to_source_similarities),
            'mean_cos_sim_real_to_source': float(np.mean(real_to_source_similarities)) if real_to_source_similarities else None,
            'std_cos_sim_real_to_source': float(np.std(real_to_source_similarities)) if real_to_source_similarities else None,
            'mean_cos_sim_fake_to_source': float(np.mean(fake_to_source_similarities)) if fake_to_source_similarities else None,
            'std_cos_sim_fake_to_source': float(np.std(fake_to_source_similarities)) if fake_to_source_similarities else None,
            'difference_real_vs_fake': (
                float(np.mean(real_to_source_similarities) - np.mean(fake_to_source_similarities))
                if real_to_source_similarities and fake_to_source_similarities else None
            ),
            'overall_linear_separability': linear_separability,  # Cross-validated accuracy (0-1)
            'num_videos_analyzed': videos_analyzed
        }
        
        return stats
    
    def plot_global_pca(self, sample_size: int = 10000, save_fig: Optional[str] = None):
        """
        Create a global PCA visualization of all embeddings colored by label.
        Uses balanced sampling of real and fake embeddings.
        
        Args:
            sample_size: Number of embedding vectors to sample for visualization (split between real/fake)
            save_fig: Optional path to save figure
        
        Returns:
            matplotlib Figure object
        """
        print(f"\n📊 Creating global PCA visualization (sampling {sample_size} embeddings, balanced)...")
        
        # Collect embeddings and labels separately for real and fake
        real_embeddings = []
        real_labels = []
        fake_embeddings = []
        fake_labels = []
        
        with h5py.File(self.data_path, 'r') as f:
            video_ids = [vid.decode() if isinstance(vid, bytes) else vid 
                        for vid in f['metadata']['video_ids'][:]]
            
            for video_id in tqdm(video_ids, desc="Collecting embeddings"):
                safe_id = video_id.replace('/', '_')
                
                try:
                    if self.use_projection is None:
                        emb = f[f'videos/{safe_id}/embeddings/{self.embedding_type}'][:]
                    else:
                        if f'videos/{safe_id}/embeddings/{self.use_projection}' not in f:
                            continue
                        emb = f[f'videos/{safe_id}/embeddings/{self.use_projection}'][:]
                    
                    labels = f[f'videos/{safe_id}/labels/audio'][:]
                    
                    # Flatten: [num_augs, num_segs, emb_dim] -> [num_augs*num_segs, emb_dim]
                    emb_flat = emb.reshape(-1, emb.shape[-1])
                    labels_flat = labels.flatten()
                    
                    # Separate real and fake
                    is_real = labels_flat == 0
                    is_fake = labels_flat > 0
                    
                    if is_real.any():
                        real_embeddings.append(emb_flat[is_real])
                        real_labels.append(labels_flat[is_real])
                    if is_fake.any():
                        fake_embeddings.append(emb_flat[is_fake])
                        fake_labels.append(labels_flat[is_fake])
                except (KeyError, ValueError):
                    continue
        
        # Get embedding dimension from first available embedding
        if len(real_embeddings) > 0:
            emb_dim = real_embeddings[0].shape[-1]
        elif len(fake_embeddings) > 0:
            emb_dim = fake_embeddings[0].shape[-1]
        else:
            raise ValueError("No embeddings found for global PCA visualization")
        
        # Stack real and fake
        if len(real_embeddings) > 0:
            real_embeddings = np.vstack(real_embeddings)
            real_labels = np.concatenate(real_labels)
        else:
            real_embeddings = np.array([]).reshape(0, emb_dim)
            real_labels = np.array([])
        
        if len(fake_embeddings) > 0:
            fake_embeddings = np.vstack(fake_embeddings)
            fake_labels = np.concatenate(fake_labels)
        else:
            fake_embeddings = np.array([]).reshape(0, emb_dim)
            fake_labels = np.array([])
        
        print(f"   Collected {len(real_embeddings):,} real and {len(fake_embeddings):,} fake embeddings")
        
        # Sample balanced amounts
        samples_per_class = sample_size // 2
        
        if len(real_embeddings) > 0:
            if len(real_embeddings) > samples_per_class:
                real_idx = np.random.choice(len(real_embeddings), samples_per_class, replace=False)
                real_sample = real_embeddings[real_idx]
                real_labels_sample = real_labels[real_idx]
            else:
                real_sample = real_embeddings
                real_labels_sample = real_labels
        else:
            real_sample = np.array([]).reshape(0, emb_dim)
            real_labels_sample = np.array([])
        
        if len(fake_embeddings) > 0:
            if len(fake_embeddings) > samples_per_class:
                fake_idx = np.random.choice(len(fake_embeddings), samples_per_class, replace=False)
                fake_sample = fake_embeddings[fake_idx]
                fake_labels_sample = fake_labels[fake_idx]
            else:
                fake_sample = fake_embeddings
                fake_labels_sample = fake_labels
        else:
            fake_sample = np.array([]).reshape(0, emb_dim)
            fake_labels_sample = np.array([])
        
        # Combine
        if len(real_sample) > 0 and len(fake_sample) > 0:
            all_embeddings = np.vstack([real_sample, fake_sample])
            all_labels = np.concatenate([real_labels_sample, fake_labels_sample])
        elif len(real_sample) > 0:
            all_embeddings = real_sample
            all_labels = real_labels_sample
        elif len(fake_sample) > 0:
            all_embeddings = fake_sample
            all_labels = fake_labels_sample
        else:
            raise ValueError("No embeddings found for global PCA visualization")
        
        print(f"   Using {len(all_embeddings):,} embeddings ({len(real_sample):,} real, {len(fake_sample):,} fake) for visualization")
        
        # Transform to PCA space
        key = self.use_projection if self.use_projection else self.embedding_type
        if key not in self.pca_models:
            raise ValueError(f"PCA not fitted for {key}. Call fit_global_pca() first.")
        
        embeddings_pca = self.pca_models[key].transform(all_embeddings)
        
        # Create visualization
        from embeddings.visualization import plot_global_pca
        fig = plot_global_pca(embeddings_pca, all_labels, self.embedding_type)
        
        if save_fig:
            fig.savefig(save_fig, dpi=300, bbox_inches='tight')
            print(f"✅ Saved global PCA visualization: {save_fig}")
        
        return fig

