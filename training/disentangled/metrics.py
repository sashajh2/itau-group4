"""
Evaluation metrics for disentangled representation learning.
Implements clustering-based metrics from Section 3.4.3.
"""
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score
)


def compute_clustering_metrics(embeddings, labels, metric='cosine', random_state=42):
    """
    Compute AMI, ARI, and Silhouette scores for embeddings.
    
    Args:
        embeddings: np.ndarray or torch.Tensor, shape [n_samples, emb_dim]
        labels: np.ndarray or torch.Tensor, shape [n_samples], binary 0=real, 1=fake
        metric: Distance metric for silhouette ('cosine' or 'euclidean')
        random_state: Random seed for k-means
    
    Returns:
        dict with keys: 'ami', 'ari', 'silhouette_gt', 'silhouette_clusters'
    """
    # Convert to numpy if needed
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Ensure labels are binary integers
    labels = labels.astype(int)
    
    # K-means clustering with k=2
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=random_state)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Compute metrics
    ami = adjusted_mutual_info_score(labels, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)
    
    # Silhouette with ground truth labels (primary metric)
    silhouette_gt = silhouette_score(embeddings, labels, metric=metric)
    
    # Silhouette with cluster labels (validation metric)
    silhouette_clusters = silhouette_score(embeddings, cluster_labels, metric=metric)
    
    return {
        'ami': float(ami),
        'ari': float(ari),
        'silhouette_gt': float(silhouette_gt),
        'silhouette_clusters': float(silhouette_clusters),
    }


def evaluate_embeddings(model, dataloader, device, use_auth=True, max_samples=10000):
    """
    Evaluate embeddings from model on dataset.
    
    Args:
        model: DisentangledProjector model
        dataloader: DataLoader for evaluation
        device: Device to run on
        use_auth: If True, use z^auth; if False, use input embeddings
        max_samples: Maximum samples to evaluate (for speed)
    
    Returns:
        dict of metrics
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            embeddings = batch['embeddings'].to(device)
            is_real = batch['is_real'].to(device)
            
            if use_auth:
                # Use z^auth from projection head
                z_auth, _ = model(embeddings)
                all_embeddings.append(z_auth.cpu())
            else:
                # Use input embeddings (before projection)
                all_embeddings.append(embeddings.cpu())
            
            # Convert is_real to 0/1 labels (0=real, 1=fake)
            labels = (~is_real).long()  # Invert: is_real=True -> label=0, is_real=False -> label=1
            all_labels.append(labels.cpu())
            
            # Limit samples for speed
            total_samples += embeddings.shape[0]
            if total_samples >= max_samples:
                break
    
    # Concatenate all batches
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Limit to max_samples exactly
    if len(all_embeddings) > max_samples:
        all_embeddings = all_embeddings[:max_samples]
        all_labels = all_labels[:max_samples]
    
    # Compute metrics
    metrics = compute_clustering_metrics(all_embeddings, all_labels, metric='cosine')
    
    return metrics

