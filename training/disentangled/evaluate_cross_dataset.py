#!/usr/bin/env python3
"""
Evaluate metrics on multiple datasets and compute cross-dataset generalization.

This script:
1. Loads embeddings from two HDF5 files (training dataset and test dataset)
2. Computes metrics on input embeddings for both datasets
3. Loads trained model checkpoint
4. Runs model to get z^auth and z^id embeddings for both datasets
5. Computes metrics on model outputs for both datasets
6. Computes cross-dataset generalization metrics (distance of test samples to training real manifold)
7. Displays comparison results
"""
import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm
import json
import os

from training.disentangled.model import DisentangledProjector
from training.disentangled.metrics import (
    compute_clustering_metrics,
    compute_distribution_metrics,
    compute_separation_metrics,
    compute_local_content_group_metrics,
)


def load_data_from_hdf5(hdf5_path: str, encoder_name: str = 'hubert', max_samples: int = None):
    """Load embeddings, labels, and content groups from HDF5 file."""
    print(f"📂 Loading data from {hdf5_path}...")
    print(f"   Encoder: {encoder_name}")
    
    all_embeddings = []
    all_labels = []
    all_content_groups = []
    content_group_map = {}
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
            
            if 'augmentation_info' not in video:
                continue
            
            source_idx = int(video['augmentation_info'].attrs.get('source_idx', 0))
            
            if f'embeddings/{encoder_name}' not in video:
                continue
            
            embeddings = video[f'embeddings/{encoder_name}'][:]
            audio_labels = video['labels/audio'][:]
            video_labels = video['labels/video'][:]
            
            num_augs, num_segs, emb_dim = embeddings.shape
            
            for aug_idx in range(num_augs):
                if max_samples and len(all_embeddings) >= max_samples:
                    break
                
                for seg_idx in range(num_segs):
                    if max_samples and len(all_embeddings) >= max_samples:
                        break
                    
                    content_group_key = (source_idx, seg_idx)
                    if content_group_key not in content_group_map:
                        content_group_map[content_group_key] = next_group_id
                        next_group_id += 1
                    
                    content_group_id = content_group_map[content_group_key]
                    
                    embedding = embeddings[aug_idx, seg_idx]
                    all_embeddings.append(embedding)
                    
                    is_real = ((audio_labels[aug_idx, seg_idx] == 0) and 
                               (video_labels[aug_idx, seg_idx] == 0))
                    label = 0 if is_real else 1
                    all_labels.append(label)
                    
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
    """Load trained model from checkpoint."""
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


def run_model_inference(model, embeddings, batch_size: int = 256, device: str = 'cuda'):
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


def compute_cross_dataset_metrics(
    train_real_embeddings,
    test_embeddings,
    train_real_z_auth,
    test_z_auth,
    name_prefix=""
):
    """
    Compute cross-dataset generalization metrics.
    
    Combines AVDeepfake1M real (label=0) + Sora2 (label=1) and uses existing metric functions.
    """
    metrics = {}
    
    # Combine embeddings: AVDeepfake1M real + Sora2
    combined_input = np.vstack([train_real_embeddings, test_embeddings])
    combined_z_auth = np.vstack([train_real_z_auth, test_z_auth])
    
    # Create labels: 0 for AVDeepfake1M real, 1 for Sora2
    combined_labels = np.concatenate([
        np.zeros(len(train_real_embeddings), dtype=int),
        np.ones(len(test_embeddings), dtype=int)
    ])
    
    # Debug output
    print(f"\n   🔍 Combined Dataset Details:")
    print(f"      Input embeddings shape: {combined_input.shape}")
    print(f"      z_auth embeddings shape: {combined_z_auth.shape}")
    print(f"      Labels: {np.sum(combined_labels == 0)} real (0) + {np.sum(combined_labels == 1)} fake (1)")
    print(f"      Unique labels: {np.unique(combined_labels)}")
    print(f"      Computing metrics on combined dataset...")
    
    # Clustering metrics (AMI, ARI, Silhouette)
    clustering_input = compute_clustering_metrics(combined_input, combined_labels, metric='cosine')
    clustering_z_auth = compute_clustering_metrics(combined_z_auth, combined_labels, metric='cosine')
    
    metrics[f'{name_prefix}ami_vs_train_real_input'] = clustering_input['ami']
    metrics[f'{name_prefix}ari_vs_train_real_input'] = clustering_input['ari']
    metrics[f'{name_prefix}silhouette_vs_train_real_input'] = clustering_input['silhouette_gt']
    
    metrics[f'{name_prefix}ami_vs_train_real_z_auth'] = clustering_z_auth['ami']
    metrics[f'{name_prefix}ari_vs_train_real_z_auth'] = clustering_z_auth['ari']
    metrics[f'{name_prefix}silhouette_vs_train_real_z_auth'] = clustering_z_auth['silhouette_gt']
    
    # Distribution metrics (KL, JS, Wasserstein)
    dist_input = compute_distribution_metrics(combined_input, combined_labels)
    dist_z_auth = compute_distribution_metrics(combined_z_auth, combined_labels)
    
    metrics[f'{name_prefix}kl_divergence_cross_dataset_input'] = dist_input['kl_divergence']
    metrics[f'{name_prefix}js_distance_cross_dataset_input'] = dist_input['js_distance']
    metrics[f'{name_prefix}wasserstein_distance_cross_dataset_input'] = dist_input['wasserstein_distance']
    
    metrics[f'{name_prefix}kl_divergence_cross_dataset_z_auth'] = dist_z_auth['kl_divergence']
    metrics[f'{name_prefix}js_distance_cross_dataset_z_auth'] = dist_z_auth['js_distance']
    metrics[f'{name_prefix}wasserstein_distance_cross_dataset_z_auth'] = dist_z_auth['wasserstein_distance']
    
    # Separation metrics (cosine similarity, distance to real manifold, etc.)
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
    
    return metrics

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate metrics on training and test datasets with cross-dataset generalization"
    )
    
    parser.add_argument('--train-hdf5', type=str,
                       default='exports/deepfake_embeddings_2.h5',
                       help='Path to training dataset HDF5 file')
    parser.add_argument('--test-hdf5', type=str,
                       default='exports/sora2_embeddings.h5',
                       help='Path to test dataset HDF5 file (Sora2)')
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
    
    print("="*100)
    print("CROSS-DATASET EVALUATION: Training (AVDeepfake1M) vs. Test (Sora2)")
    print("="*100)
    print(f"Training HDF5: {args.train_hdf5}")
    print(f"Test HDF5: {args.test_hdf5}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Encoder: {args.encoder_name}")
    print(f"Device: {device}")
    print("="*100)
    
    # Load training dataset
    print("\n" + "="*100)
    print("LOADING TRAINING DATASET (AVDeepfake1M)")
    print("="*100)
    train_embeddings, train_labels, train_content_groups = load_data_from_hdf5(
        args.train_hdf5, args.encoder_name, args.max_samples
    )
    
    # Load test dataset
    print("\n" + "="*100)
    print("LOADING TEST DATASET (Sora2)")
    print("="*100)
    test_embeddings, test_labels, test_content_groups = load_data_from_hdf5(
        args.test_hdf5, args.encoder_name, args.max_samples
    )
    
    # Separate training real and fake
    train_real_mask = train_labels == 0
    train_real_embeddings = train_embeddings[train_real_mask]
    
    # For cross-dataset evaluation, we only compute metrics on the combined dataset
    # (AVDeepfake1M real + Sora2 fake) to assess generalization
    print("\n" + "="*100)
    print("NOTE: Cross-dataset evaluation focuses on combined dataset metrics")
    print("(AVDeepfake1M real vs Sora2 fake) to assess generalization")
    print("="*100)
    
    # Initialize empty dicts for compatibility with print function
    train_input_metrics = {}
    test_input_metrics = {}
    
    # Load model
    model = load_model(args.checkpoint_path, args.input_dim, args.output_dim, device)
    
    # Run model inference
    print("\n" + "="*100)
    print("RUNNING MODEL INFERENCE")
    print("="*100)
    
    print("\n🔄 Training dataset inference...")
    train_z_auth, train_z_id = run_model_inference(
        model, train_embeddings, args.batch_size, device
    )
    
    print("\n🔄 Test dataset inference...")
    test_z_auth, test_z_id = run_model_inference(
        model, test_embeddings, args.batch_size, device
    )
    
    # Separate training real (projected)
    train_real_z_auth = train_z_auth[train_real_mask]
    
    # For cross-dataset evaluation, we only compute metrics on the combined dataset
    # Initialize empty dicts for compatibility with print function
    train_model_metrics = {}
    test_model_metrics = {}
    
    # Compute cross-dataset generalization metrics
    print("\n" + "="*100)
    print("COMPUTING CROSS-DATASET GENERALIZATION METRICS")
    print("="*100)
    
    # Show combined dataset info
    print(f"\n📊 Combined Dataset Info:")
    print(f"   AVDeepfake1M real samples: {len(train_real_embeddings):,}")
    print(f"   Sora2 samples (all fake): {len(test_embeddings):,}")
    print(f"   Total combined: {len(train_real_embeddings) + len(test_embeddings):,}")
    print(f"   Labels: {len(train_real_embeddings)} real (0) + {len(test_embeddings)} Sora2 (1)")
    
    cross_dataset_metrics = compute_cross_dataset_metrics(
        train_real_embeddings,
        test_embeddings,
        train_real_z_auth,
        test_z_auth,
        name_prefix="sora2_"
    )
    
    # Save to JSON if requested
    if args.output_json:
        output_data = {
            'train_input_metrics': train_input_metrics,
            'train_model_metrics': train_model_metrics,
            'test_input_metrics': test_input_metrics,
            'test_model_metrics': test_model_metrics,
            'cross_dataset_metrics': cross_dataset_metrics,
            'config': {
                'train_hdf5': args.train_hdf5,
                'test_hdf5': args.test_hdf5,
                'checkpoint_path': args.checkpoint_path,
                'encoder_name': args.encoder_name,
                'input_dim': args.input_dim,
                'output_dim': args.output_dim,
                'train_num_samples': len(train_embeddings),
                'test_num_samples': len(test_embeddings),
            }
        }
        
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Metrics saved to {args.output_json}")
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()

