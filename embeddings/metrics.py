"""
Metric computation functions for embedding analysis.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
from typing import Dict, Optional


def analyze_single_timestamp(embeddings: np.ndarray, labels: np.ndarray, 
                             source_idx: int) -> Dict:
    """
    Analyze embeddings across augmentations at a single timestamp.
    
    Args:
        embeddings: [num_augmentations, embedding_dim]
        labels: [num_augmentations] - audio labels
        source_idx: Index of source video
    
    Returns:
        Dictionary with metrics
    """
    is_real = labels == 0
    is_fake = labels > 0
    source_embedding = embeddings[source_idx]
    
    # Cosine similarity to source
    cos_sim = cosine_similarity(embeddings, source_embedding.reshape(1, -1)).flatten()
    
    metrics = {
        # Distance to source
        'mean_cos_sim_real_to_source': float(cos_sim[is_real].mean()) if is_real.any() else None,
        'mean_cos_sim_fake_to_source': float(cos_sim[is_fake].mean()) if is_fake.any() else None,
        'std_cos_sim_real_to_source': float(cos_sim[is_real].std()) if is_real.any() else None,
        'std_cos_sim_fake_to_source': float(cos_sim[is_fake].std()) if is_fake.any() else None,
        
        # Intra-group cohesion
        'intra_real_cohesion': pairwise_cosine_similarity(embeddings[is_real]) if is_real.sum() > 1 else None,
        'intra_fake_cohesion': pairwise_cosine_similarity(embeddings[is_fake]) if is_fake.sum() > 1 else None,
        
        # Inter-group separation
        'inter_group_separation': float(cross_group_cosine_similarity(
            embeddings[is_real], embeddings[is_fake]
        ).mean()) if is_real.any() and is_fake.any() else None,
        
        # Clustering quality
        'silhouette_score': float(silhouette_score(embeddings, labels > 0)) if len(np.unique(labels > 0)) > 1 else None,
        
        # Centroid distance
        'centroid_cosine_distance': float(1 - cosine_similarity(
            embeddings[is_real].mean(axis=0).reshape(1, -1),
            embeddings[is_fake].mean(axis=0).reshape(1, -1)
        )[0, 0]) if is_real.any() and is_fake.any() else None,
        
        # Spread
        'real_spread': float(np.linalg.norm(np.std(embeddings[is_real], axis=0))) if is_real.any() else None,
        'fake_spread': float(np.linalg.norm(np.std(embeddings[is_fake], axis=0))) if is_fake.any() else None,
        
        # Effect size
        'cohens_d': cohens_d_multivariate(embeddings[is_real], embeddings[is_fake]) if is_real.any() and is_fake.any() else None,
    }
    
    return metrics


def pairwise_cosine_similarity(embeddings: np.ndarray) -> Optional[float]:
    """Average pairwise cosine similarity within a group"""
    if len(embeddings) < 2:
        return None
    cos_sim = cosine_similarity(embeddings)
    mask = np.triu(np.ones_like(cos_sim), k=1).astype(bool)
    return float(cos_sim[mask].mean())


def cross_group_cosine_similarity(group1: np.ndarray, group2: np.ndarray) -> np.ndarray:
    """Cosine similarity between two groups"""
    return cosine_similarity(group1, group2)


def cohens_d_multivariate(group1: np.ndarray, group2: np.ndarray) -> float:
    """Effect size for multivariate data"""
    mean_diff = np.linalg.norm(group1.mean(axis=0) - group2.mean(axis=0))
    pooled_std = np.sqrt((group1.var(axis=0) + group2.var(axis=0)) / 2).mean()
    return float(mean_diff / (pooled_std + 1e-8))


def linear_classifier_accuracy(embeddings: np.ndarray, binary_labels: np.ndarray, cv: int = 3) -> Optional[float]:
    """Cross-validated linear classifier accuracy"""
    if len(np.unique(binary_labels)) < 2 or len(embeddings) < cv:
        return None
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, embeddings, binary_labels, cv=min(cv, len(embeddings)))
    return float(scores.mean())


def compute_temporal_smoothness(embeddings_seq: np.ndarray) -> Dict:
    """
    Measure how smoothly embeddings change over time.
    
    Args:
        embeddings_seq: [num_segments, embedding_dim]
    
    Returns:
        Dictionary with smoothness metrics
    """
    diffs = np.diff(embeddings_seq, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    
    return {
        'mean_step_distance': float(distances.mean()),
        'std_step_distance': float(distances.std()),
        'max_step_distance': float(distances.max()),
        'min_step_distance': float(distances.min())
    }


def label_embedding_correlation(embeddings_seq: np.ndarray, labels_seq: np.ndarray) -> Dict:
    """
    Compute correlation between label values and embedding positions.
    
    Projects embeddings onto the real-fake discriminant direction and computes
    Spearman correlation with label values.
    
    Args:
        embeddings_seq: [num_segments, embedding_dim]
        labels_seq: [num_segments] - soft labels
    
    Returns:
        Dictionary with correlation metrics
    """
    # Find real and fake means
    real_mask = labels_seq == 0
    fake_mask = labels_seq == 1
    
    if not (real_mask.any() and fake_mask.any()):
        return {'correlation': None, 'pvalue': None}
    
    mean_real = embeddings_seq[real_mask].mean(axis=0)
    mean_fake = embeddings_seq[fake_mask].mean(axis=0)
    
    # Discriminant direction
    direction = mean_fake - mean_real
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    
    # Project embeddings
    projections = embeddings_seq @ direction
    
    # Compute correlation
    corr, pval = spearmanr(projections, labels_seq)
    
    return {
        'correlation': float(corr) if not np.isnan(corr) else None,
        'pvalue': float(pval) if not np.isnan(pval) else None
    }

