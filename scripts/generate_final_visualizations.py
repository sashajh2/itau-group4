#!/usr/bin/env python3
"""
Generate comprehensive visualizations for trained disentangled models.

This script:
1. Loads input embeddings (baseline) and trained model checkpoints
2. Generates visualizations for: baseline, conservative, moderate, aggressive projections
3. Creates global plots (PCA + t-SNE) for: AVDeepfake only, AVDeepfake+ShareVeo3, AVDeepfake+ShareVeo3+Sora
4. Generates per-video experiment plots for all candidates
5. Saves results in organized directory structure: results/final_results/{model}/{projection_type}/
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Tuple, List
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import h5py

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.disentangled.model import DisentangledProjector
from training.disentangled.data_utils import load_data_from_hdf5
from training.disentangled.eval_utils import run_model_inference
from embeddings.analyzer import DeepfakeEmbeddingAnalyzer
from scripts.run_baseline_experiments import EXPERIMENT_CANDIDATES


# Model configurations
MODEL_CONFIGS = {
    'hubert': {'input_dim': 768, 'output_dim': 128},
    'openl3': {'input_dim': 512, 'output_dim': 128},
    'senet': {'input_dim': 2048, 'output_dim': 128},
}

PROJECTION_TYPES = ['baseline', 'conservative', 'moderate', 'aggressive']


def load_sora_embeddings(sora_hdf5_path: str, encoder_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Sora embeddings and labels from HDF5 file.
    
    Args:
        sora_hdf5_path: Path to Sora HDF5 file
        encoder_name: Encoder name ('hubert', 'openl3', 'senet')
    
    Returns:
        embeddings: np.ndarray [n_samples, emb_dim]
        labels: np.ndarray [n_samples] (all fake, so all 1s)
    """
    print(f"📂 Loading Sora embeddings from {sora_hdf5_path}...")
    
    all_embeddings = []
    
    with h5py.File(sora_hdf5_path, 'r') as f:
        if 'videos' not in f:
            raise ValueError(f"Sora HDF5 file must contain 'videos' group")
        
        videos_group = f['/videos']
        video_ids = list(videos_group.keys())
        
        for video_id in tqdm(video_ids, desc="Loading Sora videos"):
            video = videos_group[video_id]
            
            if f'embeddings/{encoder_name}' not in video:
                continue
            
            embeddings = video[f'embeddings/{encoder_name}'][:]  # [num_segs, emb_dim] or [num_augs, num_segs, emb_dim]
            
            # Handle different shapes
            if len(embeddings.shape) == 2:
                # [num_segs, emb_dim]
                all_embeddings.append(embeddings)
            elif len(embeddings.shape) == 3:
                # [num_augs, num_segs, emb_dim] -> flatten
                num_augs, num_segs, emb_dim = embeddings.shape
                embeddings_flat = embeddings.reshape(-1, emb_dim)
                all_embeddings.append(embeddings_flat)
    
    if len(all_embeddings) == 0:
        return np.array([]).reshape(0, 0), np.array([])
    
    embeddings = np.vstack(all_embeddings)
    labels = np.ones(len(embeddings), dtype=int)  # All fake
    
    print(f"✅ Loaded {len(embeddings):,} Sora samples")
    return embeddings, labels


def load_model_checkpoint(checkpoint_path: str, input_dim: int, output_dim: int, device: str = 'cuda') -> DisentangledProjector:
    """Load trained model from checkpoint."""
    model = DisentangledProjector(input_dim=input_dim, output_dim=output_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_data_from_hdf5_with_dataset_filter(
    hdf5_path: str,
    encoder_name: str,
    dataset_filter: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings, labels, and content groups from HDF5 file, optionally filtering by dataset.
    
    Args:
        hdf5_path: Path to HDF5 file
        encoder_name: Encoder name ('hubert', 'openl3', 'senet')
        dataset_filter: If 'avdeepfake1m', only load AVDeepfake. If 'shareveo3', only load ShareVeo3. None = all.
        max_samples: Maximum number of samples to load (None = all)
    
    Returns:
        embeddings: np.ndarray [n_samples, emb_dim]
        labels: np.ndarray [n_samples] (0=real, 1=fake)
        datasets: np.ndarray [n_samples] (dataset names)
    """
    all_embeddings = []
    all_labels = []
    all_datasets = []
    all_content_groups = []
    content_group_map = {}
    next_group_id = 0
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'videos' not in f:
            raise ValueError(f"HDF5 file must contain 'videos' group")
        
        videos_group = f['/videos']
        video_ids = list(videos_group.keys())
        
        for video_id in tqdm(video_ids, desc=f"Loading {dataset_filter or 'all'} videos"):
            if max_samples and len(all_embeddings) >= max_samples:
                break
            
            video = videos_group[video_id]
            
            # Get dataset source
            dataset = video.attrs.get('dataset', 'avdeepfake1m')
            if isinstance(dataset, bytes):
                dataset = dataset.decode()
            
            # Filter by dataset if requested
            if dataset_filter and dataset != dataset_filter:
                continue
            
            # Load metadata
            if 'augmentation_info' not in video:
                continue
            
            source_idx = int(video['augmentation_info'].attrs.get('source_idx', 0))
            
            # Load embeddings
            if f'embeddings/{encoder_name}' not in video:
                continue
            
            embeddings = video[f'embeddings/{encoder_name}'][:]  # [num_augs, num_segs, emb_dim]
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
                    
                    # Dataset
                    all_datasets.append(dataset)
                    
                    # Content group
                    all_content_groups.append(content_group_id)
    
    embeddings = np.array(all_embeddings)
    labels = np.array(all_labels, dtype=int)
    datasets = np.array(all_datasets)
    content_groups = np.array(all_content_groups, dtype=int)
    
    print(f"✅ Loaded {len(embeddings):,} samples ({dataset_filter or 'all datasets'})")
    print(f"   Real samples: {np.sum(labels == 0):,} ({100*np.sum(labels == 0)/len(labels):.1f}%)")
    print(f"   Fake samples: {np.sum(labels == 1):,} ({100*np.sum(labels == 1)/len(labels):.1f}%)")
    
    return embeddings, labels, datasets


def get_embeddings_for_projection(
    model_name: str,
    projection_type: str,
    train_hdf5: str,
    sora_hdf5: Optional[str],
    pipeline_dir: Optional[str],
    device: str = 'cuda',
    max_samples: Optional[int] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Get embeddings for a specific projection type.
    
    Returns:
        Dictionary with keys: 'avdeepfake', 'shareveo3', 'sora'
        Each value is (embeddings, labels, datasets) tuple
    """
    config = MODEL_CONFIGS[model_name]
    input_dim = config['input_dim']
    
    results = {}
    
    # Load AVDeepfake embeddings from train_hdf5
    if projection_type == 'baseline':
        # Use input embeddings directly
        embeddings, labels, datasets = load_data_from_hdf5_with_dataset_filter(
            train_hdf5, model_name, dataset_filter='avdeepfake1m', max_samples=max_samples
        )
    else:
        # Load model and run inference
        checkpoint_path = os.path.join(pipeline_dir, projection_type, 'best_model.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"🏗️  Loading {projection_type} model from {checkpoint_path}...")
        model = load_model_checkpoint(checkpoint_path, input_dim, config['output_dim'], device)
        
        # Load input embeddings
        input_embeddings, labels, datasets = load_data_from_hdf5_with_dataset_filter(
            train_hdf5, model_name, dataset_filter='avdeepfake1m', max_samples=max_samples
        )
        
        # Run inference to get z_auth
        print(f"🔄 Running inference for {projection_type} projection...")
        embeddings, _ = run_model_inference(model, input_embeddings, batch_size=256, device=device)
    
    results['avdeepfake'] = (embeddings, labels, datasets)
    
    # Load ShareVeo3 embeddings from train_hdf5
    if projection_type == 'baseline':
        sv3_embeddings, sv3_labels, sv3_datasets = load_data_from_hdf5_with_dataset_filter(
            train_hdf5, model_name, dataset_filter='shareveo3', max_samples=max_samples
        )
    else:
        # Load input embeddings and project
        sv3_input, sv3_labels, sv3_datasets = load_data_from_hdf5_with_dataset_filter(
            train_hdf5, model_name, dataset_filter='shareveo3', max_samples=max_samples
        )
        if len(sv3_input) > 0:
            sv3_embeddings, _ = run_model_inference(model, sv3_input, batch_size=256, device=device)
        else:
            sv3_embeddings = np.array([]).reshape(0, config['output_dim'])
    
    if len(sv3_embeddings) > 0:
        results['shareveo3'] = (sv3_embeddings, sv3_labels, sv3_datasets)
    
    # Load Sora embeddings if provided
    if sora_hdf5:
        if projection_type == 'baseline':
            sora_embeddings, sora_labels = load_sora_embeddings(sora_hdf5, model_name)
            sora_datasets = np.full(len(sora_embeddings), 'sora2')
        else:
            # Load Sora input embeddings
            sora_input, sora_labels = load_sora_embeddings(sora_hdf5, model_name)
            if len(sora_input) > 0:
                sora_embeddings, _ = run_model_inference(model, sora_input, batch_size=256, device=device)
                sora_datasets = np.full(len(sora_embeddings), 'sora2')
            else:
                sora_embeddings = np.array([]).reshape(0, config['output_dim'])
                sora_datasets = np.array([])
        
        results['sora'] = (sora_embeddings, sora_labels, sora_datasets)
    
    return results


def plot_global_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    datasets: np.ndarray,
    reduction_type: str,
    title_suffix: str,
    save_path: str,
    sample_size: int = 20000
):
    """
    Create global visualization using PCA or t-SNE.
    
    Args:
        embeddings: [n_samples, emb_dim]
        labels: [n_samples]
        datasets: [n_samples] dataset names
        reduction_type: 'pca' or 'tsne'
        title_suffix: Additional text for title
        save_path: Path to save figure
        sample_size: Maximum samples to use for visualization
    """
    # Sample if needed
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
        datasets = datasets[indices]
    
    # Apply dimensionality reduction
    if reduction_type == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        method_name = 'PCA'
    elif reduction_type == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embeddings_2d = reducer.fit_transform(embeddings)
        method_name = 't-SNE'
    else:
        raise ValueError(f"Unknown reduction type: {reduction_type}")
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Separate by dataset
    is_avdeepfake = datasets == 'avdeepfake1m'
    is_shareveo3 = datasets == 'shareveo3'
    is_sora = datasets == 'sora2'
    
    # Plot AVDeepfake (colored by label)
    if is_avdeepfake.any():
        avd_emb = embeddings_2d[is_avdeepfake]
        avd_labels_subset = labels[is_avdeepfake]
        scatter_avd = ax.scatter(
            avd_emb[:, 0], avd_emb[:, 1],
            c=avd_labels_subset, cmap='RdYlGn_r',
            s=20, alpha=0.6, vmin=0, vmax=1,
            edgecolors='none', label='AVDeepfake'
        )
        cbar = plt.colorbar(scatter_avd, ax=ax, label='Audio Label (0=Real, 1=Fake) - AVDeepfake', pad=0.02)
        cbar.ax.tick_params(labelsize=10)
    
    # Plot ShareVeo3 (purple)
    if is_shareveo3.any():
        sv3_emb = embeddings_2d[is_shareveo3]
        ax.scatter(
            sv3_emb[:, 0], sv3_emb[:, 1],
            c='purple', s=20, alpha=0.6,
            edgecolors='none', label='ShareVeo3'
        )
    
    # Plot Sora (orange/red)
    if is_sora.any():
        sora_emb = embeddings_2d[is_sora]
        ax.scatter(
            sora_emb[:, 0], sora_emb[:, 1],
            c='orange', s=20, alpha=0.6,
            edgecolors='none', label='Sora2'
        )
    
    ax.set_xlabel(f'{method_name}1', fontsize=12)
    ax.set_ylabel(f'{method_name}2', fontsize=12)
    
    # Build title
    title = f'Global {method_name}: {title_suffix}\n{len(embeddings):,} samples'
    counts = []
    if is_avdeepfake.any():
        counts.append(f"{np.sum(is_avdeepfake):,} AVDeepfake")
    if is_shareveo3.any():
        counts.append(f"{np.sum(is_shareveo3):,} ShareVeo3")
    if is_sora.any():
        counts.append(f"{np.sum(is_sora):,} Sora2")
    if counts:
        title += f' ({", ".join(counts)})'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {save_path}")


def generate_global_visualizations(
    model_name: str,
    projection_type: str,
    embeddings_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path
):
    """Generate all global visualizations for a projection type."""
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Generating global visualizations for {model_name}/{projection_type}...")
    
    # 1. AVDeepfake only
    if 'avdeepfake' in embeddings_dict:
        avd_emb, avd_labels, avd_datasets = embeddings_dict['avdeepfake']
        
        for reduction in ['pca', 'tsne']:
            save_path = figures_dir / f'global_{reduction}_avdeepfake_only_{model_name}_{projection_type}.png'
            plot_global_embeddings(
                avd_emb, avd_labels, avd_datasets,
                reduction, f'{model_name.upper()} - {projection_type.capitalize()}',
                str(save_path)
            )
    
    # 2. AVDeepfake + ShareVeo3
    if 'avdeepfake' in embeddings_dict and 'shareveo3' in embeddings_dict:
        avd_emb, avd_labels, avd_datasets = embeddings_dict['avdeepfake']
        sv3_emb, sv3_labels, sv3_datasets = embeddings_dict['shareveo3']
        
        # Combine
        combined_emb = np.vstack([avd_emb, sv3_emb])
        combined_labels = np.concatenate([avd_labels, sv3_labels])
        combined_datasets = np.concatenate([avd_datasets, sv3_datasets])
        
        for reduction in ['pca', 'tsne']:
            save_path = figures_dir / f'global_{reduction}_avdeepfake_shareveo3_{model_name}_{projection_type}.png'
            plot_global_embeddings(
                combined_emb, combined_labels, combined_datasets,
                reduction, f'{model_name.upper()} - {projection_type.capitalize()}',
                str(save_path)
            )
    
    # 3. AVDeepfake + ShareVeo3 + Sora
    if 'avdeepfake' in embeddings_dict and 'shareveo3' in embeddings_dict and 'sora' in embeddings_dict:
        avd_emb, avd_labels, avd_datasets = embeddings_dict['avdeepfake']
        sv3_emb, sv3_labels, sv3_datasets = embeddings_dict['shareveo3']
        sora_emb, sora_labels, sora_datasets = embeddings_dict['sora']
        
        # Combine
        combined_emb = np.vstack([avd_emb, sv3_emb, sora_emb])
        combined_labels = np.concatenate([avd_labels, sv3_labels, sora_labels])
        combined_datasets = np.concatenate([avd_datasets, sv3_datasets, sora_datasets])
        
        for reduction in ['pca', 'tsne']:
            save_path = figures_dir / f'global_{reduction}_all_datasets_{model_name}_{projection_type}.png'
            plot_global_embeddings(
                combined_emb, combined_labels, combined_datasets,
                reduction, f'{model_name.upper()} - {projection_type.capitalize()}',
                str(save_path)
            )


def generate_per_video_visualizations(
    model_name: str,
    projection_type: str,
    train_hdf5: str,
    embeddings_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
    pipeline_dir: Optional[str] = None,
    device: str = 'cuda'
):
    """Generate per-video experiment visualizations."""
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Generating per-video visualizations for {model_name}/{projection_type}...")
    
    # For baseline, use analyzer directly
    if projection_type == 'baseline':
        analyzer = DeepfakeEmbeddingAnalyzer(train_hdf5, embedding_type=model_name)
        
        # Fit global PCA for visualization
        print("   Fitting global PCA...")
        analyzer.fit_global_pca(n_components=50, sample_size=50000)
        
        # Generate plots for each experiment candidate
        for idx, (video_id, segment_index, num_fake, num_real, avg_label) in enumerate(EXPERIMENT_CANDIDATES, 1):
            print(f"   [{idx}/{len(EXPERIMENT_CANDIDATES)}] {video_id} @ segment {segment_index}")
            
            safe_video_id = video_id.replace('/', '_')
            fig_path = figures_dir / f'exp1_{model_name}_{projection_type}_{safe_video_id}_seg{segment_index}.png'
            
            try:
                metrics = analyzer.experiment1_cross_aug_single_timestamp(
                    video_id=video_id,
                    timestamp_idx=segment_index,
                    save_fig=str(fig_path)
                )
                
                if metrics is not None:
                    print(f"      ✅ Saved: {fig_path.name}")
                else:
                    print(f"      ⚠️  Skipped (single augmentation video)")
            except Exception as e:
                print(f"      ❌ Error: {e}")
                continue
    
    else:
        # For projected embeddings, we need to load model and project on-the-fly
        # This is more complex because analyzer expects HDF5 data
        print(f"   ⚠️  Per-video visualizations for projected embeddings require")
        print(f"      projected embeddings to be saved in HDF5. Skipping for now.")
        print(f"      TODO: Implement temporary HDF5 creation or modify analyzer")
        
        # TODO: Implement one of these approaches:
        # 1. Create temporary HDF5 with projected embeddings
        # 2. Modify analyzer to accept in-memory embeddings
        # 3. Create custom visualization function that works with projected embeddings directly


def process_model(
    model_name: str,
    train_hdf5: str,
    sora_hdf5: Optional[str],
    pipeline_dir: Optional[str],
    output_base_dir: Path,
    device: str = 'cuda',
    max_samples: Optional[int] = None
):
    """Process all projections for a single model."""
    print("\n" + "="*80)
    print(f"Processing Model: {model_name.upper()}")
    print("="*80)
    
    # Process each projection type
    # If pipeline_dir is provided, process all trained projections (skip baseline)
    # If pipeline_dir is None, process baseline only
    if pipeline_dir:
        projection_types_to_process = ['conservative', 'moderate', 'aggressive']
    else:
        projection_types_to_process = ['baseline']
    
    for projection_type in projection_types_to_process:
        print(f"\n{'='*80}")
        print(f"Projection Type: {projection_type.upper()}")
        print(f"{'='*80}")
        
        # New structure: {projection_type}/{model}/
        projection_output_dir = output_base_dir / projection_type / model_name
        
        try:
            # Get embeddings for this projection
            embeddings_dict = get_embeddings_for_projection(
                model_name=model_name,
                projection_type=projection_type,
                train_hdf5=train_hdf5,
                sora_hdf5=sora_hdf5,
                pipeline_dir=pipeline_dir,
                device=device,
                max_samples=max_samples
            )
            
            # Generate global visualizations
            generate_global_visualizations(
                model_name, projection_type, embeddings_dict, projection_output_dir
            )
            
            # Generate per-video visualizations
            generate_per_video_visualizations(
                model_name, projection_type, train_hdf5,
                embeddings_dict, projection_output_dir, pipeline_dir, device
            )
            
            print(f"\n✅ Completed {model_name}/{projection_type}")
            
        except Exception as e:
            print(f"\n❌ Error processing {model_name}/{projection_type}: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive visualizations for trained disentangled models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train-hdf5', type=str, required=True,
                       help='Path to training HDF5 file (contains both AVDeepfake and ShareVeo3)')
    parser.add_argument('--sora-hdf5', type=str, default=None,
                       help='Path to Sora2 HDF5 file (optional)')
    
    # Pipeline directories
    parser.add_argument('--hubert-pipeline-dir', type=str, default=None,
                       help='Path to hubert pipeline results directory (e.g., results/pipeline_sweep_001)')
    parser.add_argument('--openl3-pipeline-dir', type=str, default=None,
                       help='Path to openl3 pipeline results directory')
    parser.add_argument('--senet-pipeline-dir', type=str, default=None,
                       help='Path to senet pipeline results directory')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='results/final_results',
                       help='Output directory (default: results/final_results)')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to process (None = all)')
    
    args = parser.parse_args()
    
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("FINAL VISUALIZATION GENERATION")
    print("="*80)
    print(f"Output directory: {output_base_dir}")
    print(f"Device: {args.device}")
    print()
    
    # Process each model
    models_to_process = []
    
    if args.hubert_pipeline_dir:
        models_to_process.append(('hubert', args.hubert_pipeline_dir))
    if args.openl3_pipeline_dir:
        models_to_process.append(('openl3', args.openl3_pipeline_dir))
    if args.senet_pipeline_dir:
        models_to_process.append(('senet', args.senet_pipeline_dir))
    
    if len(models_to_process) == 0:
        print("⚠️  No pipeline directories provided. Processing baseline only for all models.")
        # Process baseline only for all models
        for model_name in ['hubert', 'openl3', 'senet']:
            process_model(
                model_name=model_name,
                train_hdf5=args.train_hdf5,
                sora_hdf5=args.sora_hdf5,
                pipeline_dir=None,
                output_base_dir=output_base_dir,
                device=args.device,
                max_samples=args.max_samples
            )
    else:
        # Process each model with its pipeline directory
        for model_name, pipeline_dir in models_to_process:
            # First process baseline for this model
            print(f"\n📊 Processing baseline for {model_name}...")
            process_model(
                model_name=model_name,
                train_hdf5=args.train_hdf5,
                sora_hdf5=args.sora_hdf5,
                pipeline_dir=None,  # Baseline doesn't need pipeline_dir
                output_base_dir=output_base_dir,
                device=args.device,
                max_samples=args.max_samples
            )
            
            # Then process trained projections (conservative, moderate, aggressive)
            print(f"\n📊 Processing trained projections for {model_name}...")
            process_model(
                model_name=model_name,
                train_hdf5=args.train_hdf5,
                sora_hdf5=args.sora_hdf5,
                pipeline_dir=pipeline_dir,
                output_base_dir=output_base_dir,
                device=args.device,
                max_samples=args.max_samples
            )
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_base_dir}/")


if __name__ == '__main__':
    main()

