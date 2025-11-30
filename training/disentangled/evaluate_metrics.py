#!/usr/bin/env python3
"""
Evaluate all metrics on input embeddings and trained model outputs.

This script:
1. Loads embeddings from HDF5 file (exports/deepfake_embeddings_2.h5)
2. Computes all metrics on input embeddings
3. Loads trained model checkpoint
4. Runs model to get z^auth and z^id embeddings
5. Computes all metrics on model outputs
6. Displays comparison results
"""
import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
import json
import os
from pathlib import Path

from training.disentangled.model import DisentangledProjector
from training.disentangled.dataset import DisentanglementDataset, disentanglement_collate_fn
from training.disentangled.metrics import (
    compute_clustering_metrics,
    compute_distribution_metrics,
    compute_separation_metrics,
    compute_local_content_group_metrics,
)


def load_data_from_hdf5(hdf5_path: str, encoder_name: str = 'hubert', max_samples: int = None):
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


def load_model(checkpoint_path: str, input_dim: int = 768, output_dim: int = 128, device: str = 'cuda'):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        input_dim: Input embedding dimension
        output_dim: Output projection dimension
        device: Device to load model on
    
    Returns:
        model: Loaded DisentangledProjector model
    """
    print(f"\n🏗️  Loading model from {checkpoint_path}...")
    
    model = DisentangledProjector(input_dim=input_dim, output_dim=output_dim)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded")
    if 'epoch' in checkpoint:
        print(f"   Trained for {checkpoint['epoch']} epochs")
    if 'val_losses' in checkpoint:
        print(f"   Best validation loss: {checkpoint['val_losses']['total']:.4f}")
    
    return model


def compute_all_metrics(embeddings, labels, content_groups=None, z_id_embeddings=None, 
                       name="Input Embeddings"):
    """
    Compute all evaluation metrics.
    
    Args:
        embeddings: np.ndarray [n_samples, emb_dim] - z^auth embeddings (or input)
        labels: np.ndarray [n_samples] - binary labels (0=real, 1=fake)
        content_groups: np.ndarray [n_samples] - content group IDs (optional)
        z_id_embeddings: np.ndarray [n_samples, emb_dim] - z^id embeddings (optional)
        name: Name for display purposes
    
    Returns:
        dict of all metrics
    """
    print(f"\n📊 Computing metrics for {name}...")
    
    metrics = {}
    
    # Clustering metrics
    print("   Computing clustering metrics...")
    clustering = compute_clustering_metrics(embeddings, labels, metric='cosine')
    metrics.update(clustering)
    
    # Distribution metrics
    print("   Computing distribution metrics...")
    distribution = compute_distribution_metrics(embeddings, labels)
    metrics.update(distribution)
    
    # Separation metrics
    print("   Computing separation metrics...")
    separation = compute_separation_metrics(embeddings, labels)
    metrics.update(separation)
    
    # Local content-group metrics (if content groups available)
    if content_groups is not None:
        print("   Computing local content-group metrics...")
        local = compute_local_content_group_metrics(
            embeddings, labels, content_groups, z_id_embeddings
        )
        metrics.update(local)
    
    return metrics


def print_metrics_comparison(input_metrics, model_metrics):
    """
    Print comparison of metrics between input and model outputs.
    
    Args:
        input_metrics: dict of metrics for input embeddings
        model_metrics: dict of metrics for model outputs
    """
    print("\n" + "="*80)
    print("METRICS COMPARISON: Input Embeddings vs. Model Outputs (z^auth)")
    print("="*80)
    
    # Group metrics by category
    clustering_keys = ['ami', 'ari', 'silhouette_gt', 'silhouette_clusters']
    distribution_keys = ['kl_divergence', 'js_distance', 'wasserstein_distance']
    separation_keys = [
        'mean_cosine_sim_real_to_real', 'mean_cosine_sim_fake_to_real',
        'separation_gap', 'mean_distance_real', 'mean_distance_fake',
        'variability_ratio', 'entropy_distance_real', 'entropy_distance_fake'
    ]
    local_keys = [
        'intra_group_cosine_sim_mean', 'intra_group_variance_real_mean',
        'intra_group_variance_fake_mean', 'intra_group_variance_ratio'
    ]
    
    def print_section(title, keys):
        print(f"\n{title}")
        print("-" * 80)
        for key in keys:
            if key in input_metrics and key in model_metrics:
                input_val = input_metrics[key]
                model_val = model_metrics[key]
                delta = model_val - input_val
                delta_pct = (delta / (abs(input_val) + 1e-10)) * 100
                print(f"  {key:40s}  Input: {input_val:8.4f}  Model: {model_val:8.4f}  "
                      f"Δ: {delta:+8.4f} ({delta_pct:+6.1f}%)")
    
    print_section("Clustering Metrics", clustering_keys)
    print_section("Distribution Metrics", distribution_keys)
    print_section("Separation Metrics", separation_keys)
    if any(k in input_metrics for k in local_keys):
        print_section("Local Content-Group Metrics", local_keys)
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all metrics on input embeddings and trained model"
    )
    
    parser.add_argument('--hdf5-path', type=str, 
                       default='exports/deepfake_embeddings_2.h5',
                       help='Path to HDF5 file with embeddings')
    parser.add_argument('--checkpoint-path', type=str,
                       default='checkpoints/disentangled_equal_weights/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--encoder-name', type=str, default='hubert',
                       choices=['hubert', 'openl3', 'senet'],
                       help='Encoder name (default: hubert)')
    parser.add_argument('--input-dim', type=int, default=768,
                       help='Input embedding dimension (default: 768 for hubert)')
    parser.add_argument('--output-dim', type=int, default=128,
                       help='Output projection dimension (default: 128)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (None = all)')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for model inference (default: 256)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--output-json', type=str, default=None,
                       help='Optional path to save metrics as JSON')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*80)
    print("EVALUATION: Input Embeddings vs. Trained Model")
    print("="*80)
    print(f"HDF5 path: {args.hdf5_path}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Encoder: {args.encoder_name}")
    print(f"Device: {device}")
    print("="*80)
    
    # Load data
    embeddings, labels, content_groups = load_data_from_hdf5(
        args.hdf5_path, args.encoder_name, args.max_samples
    )
    
    # Compute metrics on input embeddings
    input_metrics = compute_all_metrics(
        embeddings, labels, content_groups, name="Input Embeddings"
    )
    
    # Load model
    model = load_model(
        args.checkpoint_path, args.input_dim, args.output_dim, device
    )
    
    # Run model to get z^auth and z^id embeddings
    print(f"\n🔄 Running model inference...")
    model.eval()
    
    # Process in batches
    z_auth_list = []
    z_id_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(embeddings), args.batch_size), desc="Inference"):
            batch_end = min(i + args.batch_size, len(embeddings))
            batch_embeddings = torch.FloatTensor(embeddings[i:batch_end]).to(device)
            
            z_auth, z_id = model(batch_embeddings)
            
            z_auth_list.append(z_auth.cpu().numpy())
            z_id_list.append(z_id.cpu().numpy())
    
    z_auth_embeddings = np.vstack(z_auth_list)
    z_id_embeddings = np.vstack(z_id_list)
    
    print(f"✅ Generated embeddings:")
    print(f"   z^auth shape: {z_auth_embeddings.shape}")
    print(f"   z^id shape: {z_id_embeddings.shape}")
    
    # Compute metrics on model outputs
    model_metrics = compute_all_metrics(
        z_auth_embeddings, labels, content_groups, z_id_embeddings,
        name="Model Outputs (z^auth)"
    )
    
    # Print comparison
    print_metrics_comparison(input_metrics, model_metrics)
    
    # Save to JSON if requested
    if args.output_json:
        output_data = {
            'input_metrics': input_metrics,
            'model_metrics': model_metrics,
            'config': {
                'hdf5_path': args.hdf5_path,
                'checkpoint_path': args.checkpoint_path,
                'encoder_name': args.encoder_name,
                'input_dim': args.input_dim,
                'output_dim': args.output_dim,
                'num_samples': len(embeddings),
            }
        }
        
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Metrics saved to {args.output_json}")
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()

