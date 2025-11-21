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
    
    # Plot 1: Binary coloring (real=green, fake=red, source=blue circle)
    ax = axes[0]
    if is_real.any():
        ax.scatter(embeddings_pca[is_real, 0], embeddings_pca[is_real, 1],
                  c='green', label='Real', s=60, alpha=0.8, edgecolors='darkgreen', linewidths=1.5)
    if is_fake.any():
        ax.scatter(embeddings_pca[is_fake, 0], embeddings_pca[is_fake, 1],
                  c='red', label='Fake', s=60, alpha=0.8, edgecolors='darkred', linewidths=1.5)
    ax.scatter(embeddings_pca[source_idx, 0], embeddings_pca[source_idx, 1],
              c='blue', marker='o', s=80, label='Source', edgecolors='darkblue', linewidths=2, zorder=10)
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, fancybox=True, shadow=True)
    ax.set_title(f'Real vs Fake\n{video_id} @ t={timestamp:.2f}s', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Plot 2: Continuous label coloring
    ax = axes[1]
    scatter = ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
                        c=labels, cmap='RdYlGn_r', s=60, alpha=0.8,
                        edgecolors='black', linewidths=1, vmin=0, vmax=1)
    ax.scatter(embeddings_pca[source_idx, 0], embeddings_pca[source_idx, 1],
              c='blue', marker='o', s=80, label='Source', edgecolors='darkblue', linewidths=2, zorder=10)
    cbar = plt.colorbar(scatter, ax=ax, label='Audio Label', pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title('Colored by Label Value', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Plot 3: Distance to source
    ax = axes[2]
    cos_sim = cosine_similarity(embeddings_pca, embeddings_pca[source_idx].reshape(1, -1)).flatten()
    colors = ['green' if r else 'red' for r in is_real]
    bars = ax.bar(range(len(cos_sim)), cos_sim, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    if is_real.any():
        ax.axhline(y=cos_sim[is_real].mean(), color='green', linestyle='--', linewidth=2,
                  label=f'Mean Real: {cos_sim[is_real].mean():.3f}')
    if is_fake.any():
        ax.axhline(y=cos_sim[is_fake].mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean Fake: {cos_sim[is_fake].mean():.3f}')
    ax.set_xlabel('Augmentation Index', fontsize=11)
    ax.set_ylabel('Cosine Similarity to Source', fontsize=11)
    ax.set_title('Distance to Source', fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, fancybox=True, shadow=True)
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


def plot_global_pca(embeddings_pca, labels, embedding_type, datasets=None, label_type='audio'):
    """
    Create a global PCA visualization of all embeddings colored by label.
    Optionally color ShareVeo3 embeddings differently (purple).
    
    Args:
        embeddings_pca: [num_samples, 2] PCA-transformed embeddings (first 2 components)
        labels: [num_samples] labels (audio or video depending on embedding_type)
        embedding_type: Type of embedding
        datasets: Optional [num_samples] array of dataset names ('avdeepfake1m' or 'shareveo3')
        label_type: 'audio' or 'video' - determines the label name in the plot
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    if datasets is not None:
        # Color by dataset: ShareVeo3 in purple, AVDeepfake by label
        is_shareveo3 = datasets == 'shareveo3'
        is_avdeepfake = datasets == 'avdeepfake1m'
        
        # Plot AVDeepfake embeddings colored by label
        if is_avdeepfake.any():
            avd_embeddings = embeddings_pca[is_avdeepfake]
            avd_labels = labels[is_avdeepfake]
            scatter_avd = ax.scatter(
                avd_embeddings[:, 0], 
                avd_embeddings[:, 1],
                c=avd_labels, 
                cmap='RdYlGn_r', 
                s=20, 
                alpha=0.6, 
                vmin=0, 
                vmax=1,
                edgecolors='none',
                label='AVDeepfake'
            )
        
        # Plot ShareVeo3 embeddings in purple
        if is_shareveo3.any():
            sv3_embeddings = embeddings_pca[is_shareveo3]
            scatter_sv3 = ax.scatter(
                sv3_embeddings[:, 0], 
                sv3_embeddings[:, 1],
                c='purple', 
                s=20, 
                alpha=0.6, 
                edgecolors='none',
                label='ShareVeo3'
            )
        
        # Add colorbar for AVDeepfake (if present)
        if is_avdeepfake.any():
            label_name = 'Video Label' if label_type == 'video' else 'Audio Label'
            cbar = plt.colorbar(scatter_avd, ax=ax, label=f'{label_name} (0=Real, 1=Fake) - AVDeepfake', pad=0.02)
            cbar.ax.tick_params(labelsize=10)
        
        # Add legend
        ax.legend(loc='upper left', fontsize=10)
    else:
        # Original behavior: color all by label
        scatter = ax.scatter(
            embeddings_pca[:, 0], 
            embeddings_pca[:, 1],
            c=labels, 
            cmap='RdYlGn_r', 
            s=20, 
            alpha=0.6, 
            vmin=0, 
            vmax=1,
            edgecolors='none'  # No edge colors for cleaner look with many points
        )
        
        # Add colorbar
        label_name = 'Video Label' if label_type == 'video' else 'Audio Label'
        cbar = plt.colorbar(scatter, ax=ax, label=f'{label_name} (0=Real, 1=Fake)', pad=0.02)
        cbar.ax.tick_params(labelsize=10)
    
    # Labels and title
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    label_name = 'video' if label_type == 'video' else 'audio'
    title = f'Global PCA: {embedding_type.upper()} Embeddings\n{len(embeddings_pca):,} samples'
    if datasets is not None:
        avd_count = np.sum(datasets == 'avdeepfake1m') if datasets is not None else 0
        sv3_count = np.sum(datasets == 'shareveo3') if datasets is not None else 0
        title += f' ({avd_count:,} AVDeepfake, {sv3_count:,} ShareVeo3)'
    else:
        title += f' colored by {label_name} label'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add statistics text box
    real_count = np.sum(labels == 0)
    fake_count = np.sum(labels > 0)
    stats_text = f'Real (label=0): {real_count:,}\nFake (label>0): {fake_count:,}'
    if datasets is not None:
        avd_real = np.sum((labels == 0) & (datasets == 'avdeepfake1m'))
        avd_fake = np.sum((labels > 0) & (datasets == 'avdeepfake1m'))
        sv3_real = np.sum((labels == 0) & (datasets == 'shareveo3'))
        sv3_fake = np.sum((labels > 0) & (datasets == 'shareveo3'))
        stats_text += f'\n\nAVDeepfake: {avd_real:,} real, {avd_fake:,} fake'
        stats_text += f'\nShareVeo3: {sv3_real:,} real, {sv3_fake:,} fake'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

