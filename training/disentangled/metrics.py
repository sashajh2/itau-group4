"""
Evaluation metrics for disentangled representation learning.
Implements clustering-based metrics from Section 3.4.3.
Also implements distribution-based, separation, and local content-group metrics.
"""
import numpy as np
import torch
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score
)


def compute_clustering_metrics(embeddings, labels, metric='cosine', random_state=42, 
                                max_samples_for_silhouette=None):
    """
    Compute AMI, ARI, and Silhouette scores for embeddings.
    
    Args:
        embeddings: np.ndarray or torch.Tensor, shape [n_samples, emb_dim]
        labels: np.ndarray or torch.Tensor, shape [n_samples], binary 0=real, 1=fake
        metric: Distance metric for silhouette ('cosine' or 'euclidean')
        random_state: Random seed for k-means
        max_samples_for_silhouette: If provided, subsample to this many samples for silhouette
                                    computation (for speed). None = use all samples.
    
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
    
    # K-means clustering with k=2 (always on full dataset for accuracy)
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=random_state)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Compute metrics (always on full dataset)
    ami = adjusted_mutual_info_score(labels, cluster_labels)
    ari = adjusted_rand_score(labels, cluster_labels)
    
    # Silhouette computation: subsample if requested for speed
    if max_samples_for_silhouette is not None and len(embeddings) > max_samples_for_silhouette:
        # Randomly subsample for silhouette computation
        np.random.seed(random_state)
        indices = np.random.choice(len(embeddings), size=max_samples_for_silhouette, replace=False)
        silhouette_embeddings = embeddings[indices]
        silhouette_labels = labels[indices]
        silhouette_cluster_labels = cluster_labels[indices]
        
        # Use silhouette_samples for faster computation on subset
        from sklearn.metrics import silhouette_samples
        silhouette_scores_gt = silhouette_samples(silhouette_embeddings, silhouette_labels, metric=metric)
        silhouette_scores_clusters = silhouette_samples(silhouette_embeddings, silhouette_cluster_labels, metric=metric)
        
        silhouette_gt = float(np.mean(silhouette_scores_gt))
        silhouette_clusters = float(np.mean(silhouette_scores_clusters))
    else:
        # Use full dataset
        silhouette_gt = silhouette_score(embeddings, labels, metric=metric)
        silhouette_clusters = silhouette_score(embeddings, cluster_labels, metric=metric)
    
    return {
        'ami': float(ami),
        'ari': float(ari),
        'silhouette_gt': float(silhouette_gt),
        'silhouette_clusters': float(silhouette_clusters),
    }


def compute_distribution_metrics(embeddings, labels, n_bins=50):
    """
    Compute distribution-based metrics: KL divergence, JS distance, Wasserstein distance.
    
    These metrics quantify distributional differences between real and fake samples.
    Samples are projected onto one-dimensional statistic: Euclidean distance to real cluster centroid.
    
    Args:
        embeddings: np.ndarray, shape [n_samples, emb_dim]
        labels: np.ndarray, shape [n_samples], binary 0=real, 1=fake
        n_bins: Number of bins for histogram estimation
    
    Returns:
        dict with keys: 'kl_divergence', 'js_distance', 'wasserstein_distance'
    """
    # Convert to numpy if needed
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    labels = labels.astype(int)
    
    # Separate real and fake samples
    real_mask = labels == 0
    fake_mask = labels == 1
    
    real_embeddings = embeddings[real_mask]
    fake_embeddings = embeddings[fake_mask]
    
    if len(real_embeddings) == 0 or len(fake_embeddings) == 0:
        return {
            'kl_divergence': 0.0,
            'js_distance': 0.0,
            'wasserstein_distance': 0.0,
        }
    
    # Compute real cluster centroid
    real_centroid = np.mean(real_embeddings, axis=0)
    
    # Project all samples onto distance-to-centroid
    real_distances = np.linalg.norm(real_embeddings - real_centroid, axis=1)
    fake_distances = np.linalg.norm(fake_embeddings - real_centroid, axis=1)
    
    # Compute histogram bins
    all_distances = np.concatenate([real_distances, fake_distances])
    min_dist = np.min(all_distances)
    max_dist = np.max(all_distances)
    
    # Avoid edge case where all distances are the same
    if max_dist - min_dist < 1e-10:
        return {
            'kl_divergence': 0.0,
            'js_distance': 0.0,
            'wasserstein_distance': 0.0,
        }
    
    bins = np.linspace(min_dist, max_dist, n_bins + 1)
    
    # Compute histograms
    real_hist, _ = np.histogram(real_distances, bins=bins)
    fake_hist, _ = np.histogram(fake_distances, bins=bins)
    
    # Normalize to probabilities
    real_probs = real_hist.astype(float) / (np.sum(real_hist) + 1e-10)
    fake_probs = fake_hist.astype(float) / (np.sum(fake_hist) + 1e-10)
    
    # Avoid zeros for KL divergence
    real_probs = real_probs + 1e-10
    fake_probs = fake_probs + 1e-10
    real_probs = real_probs / np.sum(real_probs)
    fake_probs = fake_probs / np.sum(fake_probs)
    
    # (i) KL Divergence: D_KL(P||Q) = Σ_i p_i log(p_i/q_i)
    kl_div = np.sum(real_probs * np.log(real_probs / fake_probs))
    
    # (ii) Jensen-Shannon Distance: JS(P,Q) = sqrt((1/2)*D_KL(P||M) + (1/2)*D_KL(Q||M))
    # where M = (1/2)*(P + Q)
    js_dist = jensenshannon(real_probs, fake_probs)
    
    # (iii) Wasserstein Distance (1-Wasserstein for 1D)
    wasserstein_dist = wasserstein_distance(real_distances, fake_distances)
    
    return {
        'kl_divergence': float(kl_div),
        'js_distance': float(js_dist),
        'wasserstein_distance': float(wasserstein_dist),
    }


def compute_separation_metrics(embeddings, labels):
    """
    Compute separation metrics: mean cosine similarity and distance to real manifold.
    
    These metrics directly test the variance minimization objective and anomaly detection hypothesis.
    
    Args:
        embeddings: np.ndarray, shape [n_samples, emb_dim]
        labels: np.ndarray, shape [n_samples], binary 0=real, 1=fake
    
    Returns:
        dict with separation metrics
    """
    # Convert to numpy if needed
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    labels = labels.astype(int)
    
    # Separate real and fake samples
    real_mask = labels == 0
    fake_mask = labels == 1
    
    real_embeddings = embeddings[real_mask]
    fake_embeddings = embeddings[fake_mask]
    
    if len(real_embeddings) == 0 or len(fake_embeddings) == 0:
        return {
            'mean_cosine_sim_real_to_real': 0.0,
            'mean_cosine_sim_fake_to_real': 0.0,
            'mean_cosine_sim_real_to_fake': 0.0,
            'mean_cosine_sim_fake_to_fake': 0.0,
            'separation_gap': 0.0,
            'mean_distance_real': 0.0,
            'std_distance_real': 0.0,
            'mean_distance_fake': 0.0,
            'std_distance_fake': 0.0,
            'entropy_distance_real': 0.0,
            'entropy_distance_fake': 0.0,
            'variability_ratio': 0.0,
        }
    
    # Compute centroids
    real_centroid = np.mean(real_embeddings, axis=0)
    fake_centroid = np.mean(fake_embeddings, axis=0)
    
    # Normalize centroids
    real_centroid_norm = real_centroid / (np.linalg.norm(real_centroid) + 1e-10)
    fake_centroid_norm = fake_centroid / (np.linalg.norm(fake_centroid) + 1e-10)
    
    # (i) Mean Cosine Similarity
    # Normalize all embeddings
    real_emb_norms = np.linalg.norm(real_embeddings, axis=1, keepdims=True)
    real_emb_norms = np.where(real_emb_norms < 1e-10, 1.0, real_emb_norms)
    real_emb_normalized = real_embeddings / real_emb_norms
    
    fake_emb_norms = np.linalg.norm(fake_embeddings, axis=1, keepdims=True)
    fake_emb_norms = np.where(fake_emb_norms < 1e-10, 1.0, fake_emb_norms)
    fake_emb_normalized = fake_embeddings / fake_emb_norms
    
    # Real samples to real centroid
    sim_real_to_real = np.dot(real_emb_normalized, real_centroid_norm)
    mean_sim_real_to_real = np.mean(sim_real_to_real)
    std_sim_real_to_real = np.std(sim_real_to_real)
    
    # Fake samples to real centroid
    sim_fake_to_real = np.dot(fake_emb_normalized, real_centroid_norm)
    mean_sim_fake_to_real = np.mean(sim_fake_to_real)
    std_sim_fake_to_real = np.std(sim_fake_to_real)
    
    # Real samples to fake centroid
    sim_real_to_fake = np.dot(real_emb_normalized, fake_centroid_norm)
    mean_sim_real_to_fake = np.mean(sim_real_to_fake)
    
    # Fake samples to fake centroid
    sim_fake_to_fake = np.dot(fake_emb_normalized, fake_centroid_norm)
    mean_sim_fake_to_fake = np.mean(sim_fake_to_fake)
    
    # Separation gap: difference between real-to-real and fake-to-real similarities
    separation_gap = mean_sim_real_to_real - mean_sim_fake_to_real
    
    # (ii) Distance to Real Manifold
    # Euclidean distances to real centroid
    distances_real = np.linalg.norm(real_embeddings - real_centroid, axis=1)
    distances_fake = np.linalg.norm(fake_embeddings - real_centroid, axis=1)
    
    mean_dist_real = np.mean(distances_real)
    std_dist_real = np.std(distances_real)
    mean_dist_fake = np.mean(distances_fake)
    std_dist_fake = np.std(distances_fake)
    
    # Variability ratio
    variability_ratio = std_dist_fake / (std_dist_real + 1e-10)
    
    # Entropy of distance distributions
    def compute_entropy(distances, n_bins=20):
        """Compute entropy of distance distribution using histogram binning."""
        if len(distances) == 0:
            return 0.0
        
        hist, _ = np.histogram(distances, bins=n_bins)
        probs = hist.astype(float) / (np.sum(hist) + 1e-10)
        probs = probs[probs > 0]  # Remove zeros
        if len(probs) == 0:
            return 0.0
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy
    
    entropy_dist_real = compute_entropy(distances_real)
    entropy_dist_fake = compute_entropy(distances_fake)
    
    return {
        'mean_cosine_sim_real_to_real': float(mean_sim_real_to_real),
        'std_cosine_sim_real_to_real': float(std_sim_real_to_real),
        'mean_cosine_sim_fake_to_real': float(mean_sim_fake_to_real),
        'std_cosine_sim_fake_to_real': float(std_sim_fake_to_real),
        'mean_cosine_sim_real_to_fake': float(mean_sim_real_to_fake),
        'mean_cosine_sim_fake_to_fake': float(mean_sim_fake_to_fake),
        'separation_gap': float(separation_gap),
        'mean_distance_real': float(mean_dist_real),
        'std_distance_real': float(std_dist_real),
        'mean_distance_fake': float(mean_dist_fake),
        'std_distance_fake': float(std_dist_fake),
        'entropy_distance_real': float(entropy_dist_real),
        'entropy_distance_fake': float(entropy_dist_fake),
        'variability_ratio': float(variability_ratio),
    }


def compute_local_content_group_metrics(embeddings, labels, content_groups, z_id_embeddings=None):
    """
    Compute local content-group analysis metrics: intra-group cosine similarity and intra-group variance.
    
    These metrics validate disentanglement within individual content groups.
    
    Args:
        embeddings: np.ndarray, shape [n_samples, emb_dim] - z^auth embeddings
        labels: np.ndarray, shape [n_samples], binary 0=real, 1=fake
        content_groups: np.ndarray or list, shape [n_samples] - content group IDs
        z_id_embeddings: np.ndarray, shape [n_samples, emb_dim] - z^id embeddings (optional)
    
    Returns:
        dict with local content-group metrics
    """
    # Convert to numpy if needed
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(content_groups):
        content_groups = content_groups.cpu().numpy()
    
    labels = labels.astype(int)
    
    # Group samples by content group
    group_to_samples = defaultdict(list)
    for idx, group_id in enumerate(content_groups):
        group_to_samples[group_id].append(idx)
    
    # Filter groups with at least 2 samples
    valid_groups = {gid: indices for gid, indices in group_to_samples.items() if len(indices) >= 2}
    
    if len(valid_groups) == 0:
        return {
            'intra_group_cosine_sim_mean': 0.0,
            'intra_group_cosine_sim_std': 0.0,
            'intra_group_variance_real_mean': 0.0,
            'intra_group_variance_fake_mean': 0.0,
            'intra_group_variance_ratio': 0.0,
        }
    
    # (i) Intra-Group Cosine Similarity (for z^id if provided, else z^auth)
    # This measures whether identity features cluster tightly within content groups
    target_embeddings = z_id_embeddings if z_id_embeddings is not None else embeddings
    if torch.is_tensor(target_embeddings):
        target_embeddings = target_embeddings.cpu().numpy()
    
    intra_group_similarities = []
    
    for group_id, indices in valid_groups.items():
        group_embeddings = target_embeddings[indices]
        
        # Compute pairwise cosine similarities within group
        n_samples = len(group_embeddings)
        if n_samples < 2:
            continue
        
        # Normalize embeddings
        norms = np.linalg.norm(group_embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        normalized = group_embeddings / norms
        
        # Compute all pairwise similarities
        similarities = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                sim = np.dot(normalized[i], normalized[j])
                similarities.append(sim)
        
        if similarities:
            intra_group_similarities.extend(similarities)
    
    mean_intra_sim = np.mean(intra_group_similarities) if intra_group_similarities else 0.0
    std_intra_sim = np.std(intra_group_similarities) if intra_group_similarities else 0.0
    
    # (ii) Intra-Group Variance (for z^auth)
    # Split by authenticity label within each group
    group_variances_real = []
    group_variances_fake = []
    
    for group_id, indices in valid_groups.items():
        group_embeddings = embeddings[indices]
        group_labels = labels[indices]
        
        # Real samples in this group
        real_mask = group_labels == 0
        if np.sum(real_mask) >= 2:
            real_embeddings = group_embeddings[real_mask]
            real_mean = np.mean(real_embeddings, axis=0)
            variance_real = np.mean(np.linalg.norm(real_embeddings - real_mean, axis=1) ** 2)
            group_variances_real.append(variance_real)
        
        # Fake samples in this group
        fake_mask = group_labels == 1
        if np.sum(fake_mask) >= 2:
            fake_embeddings = group_embeddings[fake_mask]
            fake_mean = np.mean(fake_embeddings, axis=0)
            variance_fake = np.mean(np.linalg.norm(fake_embeddings - fake_mean, axis=1) ** 2)
            group_variances_fake.append(variance_fake)
    
    mean_var_real = np.mean(group_variances_real) if group_variances_real else 0.0
    mean_var_fake = np.mean(group_variances_fake) if group_variances_fake else 0.0
    
    # Variance ratio
    variance_ratio = mean_var_fake / (mean_var_real + 1e-10)
    
    return {
        'intra_group_cosine_sim_mean': float(mean_intra_sim),
        'intra_group_cosine_sim_std': float(std_intra_sim),
        'intra_group_variance_real_mean': float(mean_var_real),
        'intra_group_variance_fake_mean': float(mean_var_fake),
        'intra_group_variance_ratio': float(variance_ratio),
    }


def evaluate_embeddings(model, dataloader, device, use_auth=True, max_samples=10000, 
                        compute_all_metrics=True, max_samples_for_silhouette=None):
    """
    Evaluate embeddings from model on dataset.
    
    Args:
        model: DisentangledProjector model
        dataloader: DataLoader for evaluation
        device: Device to run on
        use_auth: If True, use z^auth; if False, use input embeddings
        max_samples: Maximum samples to evaluate (for speed)
        compute_all_metrics: If True, compute all metrics (distribution, separation, local)
        max_samples_for_silhouette: If provided, subsample to this many samples for silhouette
                                    computation (for speed). None = use all samples.
    
    Returns:
        dict of metrics
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    all_content_groups = []
    all_z_id = []
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            embeddings = batch['embeddings'].to(device)
            is_real = batch['is_real'].to(device)
            
            # Get content groups if available
            content_groups = batch.get('content_groups', None)
            
            if use_auth:
                # Use z^auth and z^id from projection head
                z_auth, z_id = model(embeddings)
                all_embeddings.append(z_auth.cpu())
                all_z_id.append(z_id.cpu())
            else:
                # Use input embeddings (before projection)
                all_embeddings.append(embeddings.cpu())
                all_z_id.append(embeddings.cpu())  # Use same for z_id if not using auth
            
            # Convert is_real to 0/1 labels (0=real, 1=fake)
            labels = (~is_real).long()  # Invert: is_real=True -> label=0, is_real=False -> label=1
            all_labels.append(labels.cpu())
            
            # Store content groups
            if content_groups is not None:
                all_content_groups.append(content_groups.cpu())
            
            # Limit samples for speed
            total_samples += embeddings.shape[0]
            if total_samples >= max_samples:
                break
    
    # Concatenate all batches
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_z_id = torch.cat(all_z_id, dim=0).numpy()
    
    # Limit to max_samples exactly
    if len(all_embeddings) > max_samples:
        all_embeddings = all_embeddings[:max_samples]
        all_labels = all_labels[:max_samples]
        all_z_id = all_z_id[:max_samples]
    
    # Compute clustering metrics (always computed)
    metrics = compute_clustering_metrics(
        all_embeddings, all_labels, metric='cosine',
        max_samples_for_silhouette=max_samples_for_silhouette
    )
    
    # Compute additional metrics if requested
    if compute_all_metrics:
        # Distribution-based metrics
        dist_metrics = compute_distribution_metrics(all_embeddings, all_labels)
        metrics.update(dist_metrics)
        
        # Separation metrics
        sep_metrics = compute_separation_metrics(all_embeddings, all_labels)
        metrics.update(sep_metrics)
        
        # Local content-group metrics (if content groups available)
        if all_content_groups:
            content_groups_array = torch.cat(all_content_groups, dim=0).numpy()
            if len(content_groups_array) > max_samples:
                content_groups_array = content_groups_array[:max_samples]
            
            local_metrics = compute_local_content_group_metrics(
                all_embeddings, all_labels, content_groups_array, all_z_id
            )
            metrics.update(local_metrics)
    
    return metrics

