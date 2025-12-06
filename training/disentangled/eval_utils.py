"""
Evaluation utilities for disentangled representation learning.
"""
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Optional, Tuple

from training.disentangled.model import DisentangledProjector
from training.disentangled.metrics import (
    compute_clustering_metrics,
    compute_distribution_metrics,
    compute_separation_metrics,
    compute_local_content_group_metrics,
)


def load_model(
    checkpoint_path: str,
    input_dim: int = 768,
    output_dim: int = 128,
    device: str = 'cuda'
) -> DisentangledProjector:
    """Load trained model from checkpoint."""
    model = DisentangledProjector(input_dim=input_dim, output_dim=output_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def run_model_inference(
    model: DisentangledProjector,
    embeddings: np.ndarray,
    batch_size: int = 256,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model inference to get z^auth and z^id embeddings."""
    model.eval()
    z_auth_list = []
    z_id_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(embeddings), batch_size), desc="Inference"):
            batch_end = min(i + batch_size, len(embeddings))
            batch_embeddings = torch.FloatTensor(embeddings[i:batch_end]).to(device)
            
            z_auth, z_id = model(batch_embeddings)
            
            z_auth_list.append(z_auth.cpu().numpy())
            z_id_list.append(z_id.cpu().numpy())
    
    z_auth_embeddings = np.vstack(z_auth_list)
    z_id_embeddings = np.vstack(z_id_list)
    
    return z_auth_embeddings, z_id_embeddings


def evaluate_single_dataset(
    embeddings: np.ndarray,
    labels: np.ndarray,
    content_groups: np.ndarray,
    checkpoint_path: str,
    input_dim: int = 768,
    output_dim: int = 128,
    batch_size: int = 256,
    device: str = 'cuda',
    verbose: bool = True,
    max_samples_for_silhouette: int = 5000
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on single dataset (matches evaluate_metrics.py).
    
    Includes ALL metrics: clustering, distribution, separation, and local content-group metrics.
    
    Args:
        embeddings: Input embeddings [n_samples, emb_dim]
        labels: Labels [n_samples] (0=real, 1=fake)
        content_groups: Content groups [n_samples]
        checkpoint_path: Path to model checkpoint
        input_dim: Input embedding dimension
        output_dim: Output projection dimension
        batch_size: Batch size for inference
        device: Device to use
        verbose: Whether to print progress
    
    Returns:
        {
            'input_metrics': {...},  # Metrics on input embeddings
            'model_metrics': {...}   # Metrics on model outputs (z^auth)
        }
    """
    if verbose:
        print(f"\n📊 Evaluating single dataset...")
        print(f"   Samples: {len(embeddings):,}")
        print(f"   Real: {np.sum(labels == 0):,}, Fake: {np.sum(labels == 1):,}")
    
    # Compute metrics on input embeddings
    if verbose:
        print("\n   Computing metrics on input embeddings...")
    
    input_metrics = {}
    
    # Clustering metrics
    if verbose:
        print("      Computing clustering metrics...")
    clustering = compute_clustering_metrics(
        embeddings, labels, metric='cosine',
        max_samples_for_silhouette=max_samples_for_silhouette
    )
    input_metrics.update(clustering)
    
    # Distribution metrics
    if verbose:
        print("      Computing distribution metrics...")
    distribution = compute_distribution_metrics(embeddings, labels)
    input_metrics.update(distribution)
    
    # Separation metrics
    if verbose:
        print("      Computing separation metrics...")
    separation = compute_separation_metrics(embeddings, labels)
    input_metrics.update(separation)
    
    # Local content-group metrics (INCLUDED for single dataset)
    if content_groups is not None:
        if verbose:
            print("      Computing local content-group metrics...")
        local = compute_local_content_group_metrics(
            embeddings, labels, content_groups, z_id_embeddings=None
        )
        input_metrics.update(local)
    
    # Load model and run inference
    if verbose:
        print("\n   Loading model and running inference...")
    model = load_model(checkpoint_path, input_dim, output_dim, device)
    z_auth_embeddings, z_id_embeddings = run_model_inference(
        model, embeddings, batch_size, device
    )
    
    # Compute metrics on model outputs
    if verbose:
        print("\n   Computing metrics on model outputs (z^auth)...")
    
    model_metrics = {}
    
    # Clustering metrics
    if verbose:
        print("      Computing clustering metrics...")
    clustering = compute_clustering_metrics(
        z_auth_embeddings, labels, metric='cosine',
        max_samples_for_silhouette=max_samples_for_silhouette
    )
    model_metrics.update(clustering)
    
    # Distribution metrics
    if verbose:
        print("      Computing distribution metrics...")
    distribution = compute_distribution_metrics(z_auth_embeddings, labels)
    model_metrics.update(distribution)
    
    # Separation metrics
    if verbose:
        print("      Computing separation metrics...")
    separation = compute_separation_metrics(z_auth_embeddings, labels)
    model_metrics.update(separation)
    
    # Local content-group metrics (INCLUDED for single dataset)
    if content_groups is not None:
        if verbose:
            print("      Computing local content-group metrics...")
        local = compute_local_content_group_metrics(
            z_auth_embeddings, labels, content_groups, z_id_embeddings
        )
        model_metrics.update(local)
    
    return {
        'input_metrics': input_metrics,
        'model_metrics': model_metrics,
    }


def evaluate_cross_dataset(
    train_real_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    checkpoint_path: str,
    input_dim: int = 768,
    output_dim: int = 128,
    batch_size: int = 256,
    device: str = 'cuda',
    verbose: bool = True,
    max_samples_for_silhouette: int = 5000
) -> Dict[str, float]:
    """
    Evaluate cross-dataset generalization (matches evaluate_cross_dataset.py).
    
    Does NOT include local content-group metrics (only clustering, distribution, separation).
    
    Args:
        train_real_embeddings: Training real samples only [n_train_real, emb_dim]
        test_embeddings: Test samples [n_test, emb_dim] (all fake for Sora2)
        checkpoint_path: Path to model checkpoint
        input_dim: Input embedding dimension
        output_dim: Output projection dimension
        batch_size: Batch size for inference
        device: Device to use
        verbose: Whether to print progress
    
    Returns:
        {
            'cross_dataset_ami_input': ...,
            'cross_dataset_ami_z_auth': ...,
            'cross_dataset_ari_input': ...,
            'cross_dataset_ari_z_auth': ...,
            'cross_dataset_silhouette_gt_input': ...,
            'cross_dataset_silhouette_gt_z_auth': ...,
            'cross_dataset_kl_divergence_input': ...,
            'cross_dataset_kl_divergence_z_auth': ...,
            'cross_dataset_js_distance_input': ...,
            'cross_dataset_js_distance_z_auth': ...,
            'cross_dataset_wasserstein_distance_input': ...,
            'cross_dataset_wasserstein_distance_z_auth': ...,
            'cross_dataset_separation_gap_input': ...,
            'cross_dataset_separation_gap_z_auth': ...,
            'cross_dataset_mean_distance_real_input': ...,
            'cross_dataset_mean_distance_real_z_auth': ...,
            'cross_dataset_mean_distance_fake_input': ...,
            'cross_dataset_mean_distance_fake_z_auth': ...,
            'cross_dataset_variability_ratio_input': ...,
            'cross_dataset_variability_ratio_z_auth': ...,
        }
    """
    if verbose:
        print(f"\n📊 Evaluating cross-dataset generalization...")
        print(f"   Train real samples: {len(train_real_embeddings):,}")
        print(f"   Test samples: {len(test_embeddings):,}")
    
    # Load model and run inference
    if verbose:
        print("\n   Loading model and running inference...")
    model = load_model(checkpoint_path, input_dim, output_dim, device)
    
    train_real_z_auth, _ = run_model_inference(
        model, train_real_embeddings, batch_size, device
    )
    test_z_auth, _ = run_model_inference(
        model, test_embeddings, batch_size, device
    )
    
    # Combine embeddings: AVDeepfake1M real + Sora2
    combined_input = np.vstack([train_real_embeddings, test_embeddings])
    combined_z_auth = np.vstack([train_real_z_auth, test_z_auth])
    
    # Create labels: 0 for AVDeepfake1M real, 1 for Sora2
    combined_labels = np.concatenate([
        np.zeros(len(train_real_embeddings), dtype=int),
        np.ones(len(test_embeddings), dtype=int)
    ])
    
    if verbose:
        print(f"\n   🔍 Combined Dataset Details:")
        print(f"      Input embeddings shape: {combined_input.shape}")
        print(f"      z_auth embeddings shape: {combined_z_auth.shape}")
        print(f"      Labels: {np.sum(combined_labels == 0)} real (0) + {np.sum(combined_labels == 1)} fake (1)")
        print(f"      Computing metrics on combined dataset...")
    
    metrics = {}
    
    # Clustering metrics (AMI, ARI, Silhouette)
    if verbose:
        print("      Computing clustering metrics...")
    clustering_input = compute_clustering_metrics(
        combined_input, combined_labels, metric='cosine',
        max_samples_for_silhouette=max_samples_for_silhouette
    )
    clustering_z_auth = compute_clustering_metrics(
        combined_z_auth, combined_labels, metric='cosine',
        max_samples_for_silhouette=max_samples_for_silhouette
    )
    
    # Use empty prefix to match evaluate_cross_dataset.py when called without prefix
    name_prefix = ""
    
    metrics[f'{name_prefix}ami_vs_train_real_input'] = clustering_input['ami']
    metrics[f'{name_prefix}ari_vs_train_real_input'] = clustering_input['ari']
    metrics[f'{name_prefix}silhouette_vs_train_real_input'] = clustering_input['silhouette_gt']
    
    metrics[f'{name_prefix}ami_vs_train_real_z_auth'] = clustering_z_auth['ami']
    metrics[f'{name_prefix}ari_vs_train_real_z_auth'] = clustering_z_auth['ari']
    metrics[f'{name_prefix}silhouette_vs_train_real_z_auth'] = clustering_z_auth['silhouette_gt']
    
    # Distribution metrics (KL, JS, Wasserstein)
    if verbose:
        print("      Computing distribution metrics...")
    dist_input = compute_distribution_metrics(combined_input, combined_labels)
    dist_z_auth = compute_distribution_metrics(combined_z_auth, combined_labels)
    
    metrics[f'{name_prefix}kl_divergence_cross_dataset_input'] = dist_input['kl_divergence']
    metrics[f'{name_prefix}js_distance_cross_dataset_input'] = dist_input['js_distance']
    metrics[f'{name_prefix}wasserstein_distance_cross_dataset_input'] = dist_input['wasserstein_distance']
    
    metrics[f'{name_prefix}kl_divergence_cross_dataset_z_auth'] = dist_z_auth['kl_divergence']
    metrics[f'{name_prefix}js_distance_cross_dataset_z_auth'] = dist_z_auth['js_distance']
    metrics[f'{name_prefix}wasserstein_distance_cross_dataset_z_auth'] = dist_z_auth['wasserstein_distance']
    
    # Separation metrics (cosine similarity, distance to real manifold, etc.)
    if verbose:
        print("      Computing separation metrics...")
    sep_input = compute_separation_metrics(combined_input, combined_labels)
    sep_z_auth = compute_separation_metrics(combined_z_auth, combined_labels)
    
    # Add key separation metrics
    metrics[f'{name_prefix}separation_gap_cross_dataset_input'] = sep_input['separation_gap']
    metrics[f'{name_prefix}mean_distance_real_cross_dataset_input'] = sep_input['mean_distance_real']
    metrics[f'{name_prefix}mean_distance_fake_cross_dataset_input'] = sep_input['mean_distance_fake']
    metrics[f'{name_prefix}variability_ratio_cross_dataset_input'] = sep_input['variability_ratio']
    
    metrics[f'{name_prefix}separation_gap_cross_dataset_z_auth'] = sep_z_auth['separation_gap']
    metrics[f'{name_prefix}mean_distance_real_cross_dataset_z_auth'] = sep_z_auth['mean_distance_real']
    metrics[f'{name_prefix}mean_distance_fake_cross_dataset_z_auth'] = sep_z_auth['mean_distance_fake']
    metrics[f'{name_prefix}variability_ratio_cross_dataset_z_auth'] = sep_z_auth['variability_ratio']
    
    # NOTE: NO local content-group metrics for cross-dataset evaluation
    
    return metrics

