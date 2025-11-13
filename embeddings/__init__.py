"""
Embedding analysis tools for deepfake detection.

This package provides:
- DeepfakeEmbeddingAnalyzer: Main analyzer class for embedding analysis
- Metrics: Functions for computing embedding metrics
- Visualization: Plotting functions for embedding analysis
"""

from .analyzer import DeepfakeEmbeddingAnalyzer
from .metrics import (
    analyze_single_timestamp,
    compute_temporal_smoothness,
    label_embedding_correlation,
    pairwise_cosine_similarity,
    cross_group_cosine_similarity,
    cohens_d_multivariate,
    linear_classifier_accuracy,
)

__all__ = [
    "DeepfakeEmbeddingAnalyzer",
    "analyze_single_timestamp",
    "compute_temporal_smoothness",
    "label_embedding_correlation",
    "pairwise_cosine_similarity",
    "cross_group_cosine_similarity",
    "cohens_d_multivariate",
    "linear_classifier_accuracy",
]

