"""
Visualization functions for embedding analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def plot_cross_augmentation_timestamp(embeddings_pca, labels, source_idx,
                                      video_id, timestamp, embedding_type):
    """
    Create 3-panel figure for cross-augmentation analysis.
    
    Args:
        embeddings_pca: [num_augmentations, n_components] PCA-transformed embeddings
        labels: [num_augmentations] audio labels
        source_idx: Index of source video
        video_id: Video identifier
        timestamp: Timestamp in seconds
        embedding_type: Type of embedding
    
    Returns:
        matplotlib Figure object
    """
    is_real = labels == 0
    is_fake = labels > 0
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Binary coloring (real=green, fake=red, source=blue star)
    ax = axes[0]
    if is_real.any():
        ax.scatter(embeddings_pca[is_real, 0], embeddings_pca[is_real, 1],
                  c='green', label='Real', s=100, alpha=0.6, edgecolors='black')
    if is_fake.any():
        ax.scatter(embeddings_pca[is_fake, 0], embeddings_pca[is_fake, 1],
                  c='red', label='Fake', s=100, alpha=0.6, edgecolors='black')
    ax.scatter(embeddings_pca[source_idx, 0], embeddings_pca[source_idx, 1],
              c='blue', marker='*', s=500, label='Source', edgecolors='black', linewidths=2)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.set_title(f'Real vs Fake\n{video_id} @ t={timestamp:.2f}s')
    ax.grid(alpha=0.3)
    
    # Plot 2: Continuous label coloring
    ax = axes[1]
    scatter = ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
                        c=labels, cmap='RdYlGn_r', s=100, alpha=0.7,
                        edgecolors='black', vmin=0, vmax=1)
    ax.scatter(embeddings_pca[source_idx, 0], embeddings_pca[source_idx, 1],
              c='blue', marker='*', s=500, label='Source', edgecolors='black', linewidths=2)
    plt.colorbar(scatter, ax=ax, label='Audio Label')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Colored by Label Value')
    ax.grid(alpha=0.3)
    
    # Plot 3: Distance to source
    ax = axes[2]
    cos_sim = cosine_similarity(embeddings_pca, embeddings_pca[source_idx].reshape(1, -1)).flatten()
    colors = ['green' if r else 'red' for r in is_real]
    bars = ax.bar(range(len(cos_sim)), cos_sim, color=colors, alpha=0.6, edgecolor='black')
    if is_real.any():
        ax.axhline(y=cos_sim[is_real].mean(), color='green', linestyle='--',
                  label=f'Mean Real: {cos_sim[is_real].mean():.3f}')
    if is_fake.any():
        ax.axhline(y=cos_sim[is_fake].mean(), color='red', linestyle='--',
                  label=f'Mean Fake: {cos_sim[is_fake].mean():.3f}')
    ax.set_xlabel('Augmentation Index')
    ax.set_ylabel('Cosine Similarity to Source')
    ax.set_title('Distance to Source')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'Cross-Augmentation Analysis: {embedding_type}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_single_video_temporal(embeddings_pca, labels_seq, video_path):
    """
    Create 2-panel figure for temporal analysis.
    
    Args:
        embeddings_pca: [num_segments, n_components] PCA-transformed embeddings
        labels_seq: [num_segments] audio labels
        video_path: Path to video file
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Trajectory in PCA space
    ax = axes[0]
    scatter = ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
                        c=labels_seq, cmap='RdYlGn_r', s=50, alpha=0.7, vmin=0, vmax=1)
    ax.plot(embeddings_pca[:, 0], embeddings_pca[:, 1], 'k-', alpha=0.3, linewidth=1)
    plt.colorbar(scatter, ax=ax, label='Audio Label (Fakeness)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'Temporal Trajectory\n{video_path.split("/")[-1]}')
    ax.grid(alpha=0.3)
    
    # Plot 2: Labels over time
    ax = axes[1]
    ax.plot(labels_seq, 'o-', color='darkblue', markersize=4)
    ax.set_xlabel('Segment Index')
    ax.set_ylabel('Audio Label')
    ax.set_title('Label Sequence')
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

