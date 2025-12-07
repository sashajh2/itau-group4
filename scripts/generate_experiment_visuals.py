#!/usr/bin/env python3
"""
Generate visualizations for specific experiment candidates.

This script focuses solely on generating cross-augmentation visualizations
for the EXPERIMENT_CANDIDATES list from run_baseline_experiments.py.

Supports both baseline embeddings and trained projections (conservative, moderate, aggressive).
"""

import os
import sys
import argparse
import numpy as np
import torch
import h5py
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.analyzer import DeepfakeEmbeddingAnalyzer
from embeddings.visualization import plot_cross_augmentation_timestamp
from embeddings.metrics import analyze_single_timestamp
from training.disentangled.eval_utils import run_model_inference
from training.disentangled.model import DisentangledProjector

# Query 3 results - top 20 candidates
# Format: (video_id, segment_index, num_fake, num_real, avg_label)
EXPERIMENT_CANDIDATES = [
    ('gqpErbFnbiY/00017', 8, 15, 5, 0.750),
    ('gqpErbFnbiY/00015', 13, 15, 5, 0.750),
    ('gqpErbFnbiY/00015', 14, 15, 5, 0.750),
    ('gqpErbFnbiY/00017', 7, 15, 5, 0.750),
    ('gqpErbFnbiY/00017', 9, 15, 5, 0.750),
    ('gqpErbFnbiY/00016', 43, 10, 10, 0.500),
    ('gdg4mUSwuhI/00002', 14, 10, 10, 0.500),
    ('gqpErbFnbiY/00002', 23, 10, 10, 0.500),
    ('gqpErbFnbiY/00025', 4, 10, 9, 0.526),
    ('gqpErbFnbiY/00015', 6, 10, 10, 0.500),
    ('golS4kh8ETY/00002', 8, 10, 10, 0.500),
    ('gqpErbFnbiY/00026', 35, 10, 10, 0.500),
    ('gjCwsdCssdk/00019', 30, 10, 10, 0.500),
    ('gqpErbFnbiY/00022', 10, 10, 10, 0.500),
    ('gqpErbFnbiY/00018', 5, 10, 10, 0.500),
    ('gqpErbFnbiY/00007', 21, 10, 9, 0.526),
    ('gqpErbFnbiY/00008', 71, 10, 9, 0.526),
    ('gqpErbFnbiY/00010', 28, 10, 10, 0.500),
    ('gqpErbFnbiY/00008', 68, 10, 9, 0.526),
    ('gqpErbFnbiY/00016', 41, 10, 10, 0.500),
]


# Model configurations
MODEL_CONFIGS = {
    'hubert': {'input_dim': 768, 'output_dim': 128},
    'openl3': {'input_dim': 512, 'output_dim': 128},
    'senet': {'input_dim': 2048, 'output_dim': 128},
}


def generate_visuals_baseline(embedding_type: str, data_path: str, output_dir: Path):
    """
    Generate visualizations for baseline embeddings.
    
    Args:
        embedding_type: 'hubert', 'openl3', or 'senet'
        data_path: Path to HDF5 data file
        output_dir: Output directory for figures
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Generating Experiment Visualizations: {embedding_type.upper()} (BASELINE)")
    print("=" * 80)
    print(f"Data file: {data_path}")
    print(f"Embedding type: {embedding_type}")
    print(f"Output directory: {output_dir}")
    print(f"Number of candidates: {len(EXPERIMENT_CANDIDATES)}")
    print()
    
    # Initialize analyzer
    print("📊 Initializing analyzer...")
    analyzer = DeepfakeEmbeddingAnalyzer(data_path, embedding_type=embedding_type)
    print(f"   Loaded metadata: {analyzer.data['total_videos']} videos")
    print()
    
    # Fit PCA (required for visualization)
    print("🔧 Fitting global PCA...")
    analyzer.fit_global_pca(n_components=50, sample_size=50000)
    print("   ✅ PCA fitted successfully")
    print()
    
    # Generate visualizations for each candidate
    print("=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for idx, (video_id, segment_index, num_fake, num_real, avg_label) in enumerate(EXPERIMENT_CANDIDATES, 1):
        print(f"\n[{idx}/{len(EXPERIMENT_CANDIDATES)}] {video_id} @ segment {segment_index}")
        print(f"   Expected: {num_fake} fake, {num_real} real augmentations, avg_label={avg_label:.3f}")
        
        # Create safe filename
        safe_video_id = video_id.replace('/', '_')
        fig_path = output_dir / f'exp1_{embedding_type}_baseline_{safe_video_id}_seg{segment_index}.png'
        
        try:
            # Generate visualization
            metrics = analyzer.experiment1_cross_aug_single_timestamp(
                video_id=video_id,
                timestamp_idx=segment_index,
                save_fig=str(fig_path)
            )
            
            if metrics is not None:
                success_count += 1
                print(f"   ✅ Saved: {fig_path.name}")
                print(f"      Silhouette score: {metrics.get('silhouette_score', 'N/A'):.4f}" if metrics.get('silhouette_score') is not None else "      Silhouette score: N/A")
                print(f"      Centroid distance: {metrics.get('centroid_cosine_distance', 'N/A'):.4f}" if metrics.get('centroid_cosine_distance') is not None else "      Centroid distance: N/A")
            else:
                skip_count += 1
                print(f"   ⚠️  Skipped (single augmentation video)")
                
        except Exception as e:
            error_count += 1
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"   ✅ Successfully generated: {success_count}/{len(EXPERIMENT_CANDIDATES)}")
    print(f"   ⚠️  Skipped: {skip_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"\n   Figures saved to: {output_dir}")


def generate_visuals_projected(
    embedding_type: str, 
    projection_type: str,
    data_path: str, 
    model_path: str,
    output_dir: Path,
    device: str = 'cuda'
):
    """
    Generate visualizations for projected embeddings (conservative, moderate, aggressive).
    
    Args:
        embedding_type: 'hubert', 'openl3', or 'senet'
        projection_type: 'conservative', 'moderate', or 'aggressive'
        data_path: Path to HDF5 data file
        model_path: Path to model checkpoint
        output_dir: Output directory for figures
        device: Device to use for inference
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Generating Experiment Visualizations: {embedding_type.upper()}/{projection_type.upper()}")
    print("=" * 80)
    print(f"Data file: {data_path}")
    print(f"Embedding type: {embedding_type}")
    print(f"Projection type: {projection_type}")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of candidates: {len(EXPERIMENT_CANDIDATES)}")
    print()
    
    # Load model
    config = MODEL_CONFIGS[embedding_type]
    print(f"🏗️  Loading {projection_type} model...")
    # Load checkpoint to CPU first to avoid device validation issues on Mac
    model = DisentangledProjector(input_dim=config['input_dim'], output_dim=config['output_dim'])
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("   ✅ Model loaded successfully")
    print()
    
    # Fit PCA on a sample of projected embeddings
    print("🔧 Fitting global PCA on projected embeddings...")
    pca = fit_pca_on_projected_embeddings(data_path, embedding_type, model, config, device)
    print("   ✅ PCA fitted successfully")
    print()
    
    # Generate visualizations for each candidate
    print("=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for idx, (video_id, segment_index, num_fake, num_real, avg_label) in enumerate(EXPERIMENT_CANDIDATES, 1):
        print(f"\n[{idx}/{len(EXPERIMENT_CANDIDATES)}] {video_id} @ segment {segment_index}")
        print(f"   Expected: {num_fake} fake, {num_real} real augmentations, avg_label={avg_label:.3f}")
        
        # Create safe filename
        safe_video_id = video_id.replace('/', '_')
        fig_path = output_dir / f'exp1_{embedding_type}_{projection_type}_{safe_video_id}_seg{segment_index}.png'
        
        try:
            # Load video data and get embeddings at this timestamp
            video_data = load_video_data(data_path, video_id, embedding_type)
            
            # Check if video has multiple augmentations
            if video_data['num_augmentations'] == 1:
                skip_count += 1
                print(f"   ⚠️  Skipped (single augmentation video)")
                continue
            
            # Check timestamp index
            if segment_index >= video_data['num_segments']:
                error_count += 1
                print(f"   ❌ Error: segment_index {segment_index} >= num_segments {video_data['num_segments']}")
                continue
            
            # Extract embeddings at this timestamp: [num_augs, emb_dim]
            input_embeddings = video_data['embeddings'][:, segment_index, :]
            labels = video_data['labels']['audio'][:, segment_index]
            source_idx = video_data['augmentation_info']['source_idx']
            
            # Run inference to get projected embeddings
            z_auth, _ = run_model_inference(model, input_embeddings, batch_size=256, device=device)
            
            # Transform to PCA space
            embeddings_pca = pca.transform(z_auth)
            
            # Compute metrics
            metrics = analyze_single_timestamp(z_auth, labels, source_idx)
            
            # Create visualization
            segment_duration = 0.15  # Default segment duration
            fig = plot_cross_augmentation_timestamp(
                embeddings_pca, labels, source_idx,
                video_id, segment_index * segment_duration,
                f"{embedding_type}_{projection_type}"
            )
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            success_count += 1
            print(f"   ✅ Saved: {fig_path.name}")
            print(f"      Silhouette score: {metrics.get('silhouette_score', 'N/A'):.4f}" if metrics.get('silhouette_score') is not None else "      Silhouette score: N/A")
            print(f"      Centroid distance: {metrics.get('centroid_cosine_distance', 'N/A'):.4f}" if metrics.get('centroid_cosine_distance') is not None else "      Centroid distance: N/A")
                
        except Exception as e:
            error_count += 1
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"   ✅ Successfully generated: {success_count}/{len(EXPERIMENT_CANDIDATES)}")
    print(f"   ⚠️  Skipped: {skip_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"\n   Figures saved to: {output_dir}")


def load_video_data(data_path: str, video_id: str, embedding_type: str) -> dict:
    """Load video data from HDF5."""
    safe_id = video_id.replace('/', '_')
    
    with h5py.File(data_path, 'r') as f:
        if safe_id not in f['videos']:
            raise ValueError(f"Video {video_id} not found in HDF5 file")
        
        vid_grp = f['videos'][safe_id]
        
        # Load metadata
        data = {
            'dataset': vid_grp.attrs['dataset'].decode() if isinstance(vid_grp.attrs['dataset'], bytes) else vid_grp.attrs['dataset'],
            'num_segments': int(vid_grp.attrs['num_segments']),
            'num_augmentations': int(vid_grp.attrs['num_augmentations']),
            'augmentation_info': {
                'video_paths': [p.decode() if isinstance(p, bytes) else p 
                               for p in vid_grp['augmentation_info/video_paths'][:]],
                'types': [t.decode() if isinstance(t, bytes) else t 
                         for t in vid_grp['augmentation_info/types'][:]],
                'source_idx': int(vid_grp['augmentation_info'].attrs['source_idx'])
            },
            'labels': {
                'audio': vid_grp['labels/audio'][:],
                'video': vid_grp['labels/video'][:]
            }
        }
        
        # Load embeddings
        data['embeddings'] = vid_grp[f'embeddings/{embedding_type}'][:]
        
        return data


def fit_pca_on_projected_embeddings(
    data_path: str,
    embedding_type: str,
    model,
    config: dict,
    device: str,
    sample_size: int = 50000
) -> PCA:
    """Fit PCA on a sample of projected embeddings."""
    print(f"   Collecting {sample_size} samples for PCA fitting...")
    
    real_embeddings = []
    fake_embeddings = []
    
    with h5py.File(data_path, 'r') as f:
        video_ids = [vid.decode() if isinstance(vid, bytes) else vid 
                    for vid in f['metadata']['video_ids'][:]]
        
        for video_id in video_ids[:500]:  # Limit to first 500 videos for speed
            safe_id = video_id.replace('/', '_')
            
            try:
                if f'videos/{safe_id}/embeddings/{embedding_type}' not in f:
                    continue
                
                emb = f[f'videos/{safe_id}/embeddings/{embedding_type}'][:]
                labels = f[f'videos/{safe_id}/labels/audio'][:]
                
                # Flatten: [num_augs, num_segs, emb_dim] -> [num_augs*num_segs, emb_dim]
                emb_flat = emb.reshape(-1, emb.shape[-1])
                labels_flat = labels.flatten()
                
                # Separate real and fake
                is_real = labels_flat == 0
                is_fake = labels_flat > 0
                
                if is_real.any() and len(real_embeddings) * len(emb_flat[is_real]) < sample_size // 2:
                    real_embeddings.append(emb_flat[is_real])
                if is_fake.any() and len(fake_embeddings) * len(emb_flat[is_fake]) < sample_size // 2:
                    fake_embeddings.append(emb_flat[is_fake])
                    
            except KeyError:
                continue
    
    # Stack and sample
    if len(real_embeddings) > 0:
        real_embeddings = np.vstack(real_embeddings)
    else:
        real_embeddings = np.array([]).reshape(0, config['input_dim'])
    
    if len(fake_embeddings) > 0:
        fake_embeddings = np.vstack(fake_embeddings)
    else:
        fake_embeddings = np.array([]).reshape(0, config['input_dim'])
    
    # Sample balanced amounts
    samples_per_class = sample_size // 2
    if len(real_embeddings) > samples_per_class:
        real_idx = np.random.choice(len(real_embeddings), samples_per_class, replace=False)
        real_sample = real_embeddings[real_idx]
    else:
        real_sample = real_embeddings
    
    if len(fake_embeddings) > samples_per_class:
        fake_idx = np.random.choice(len(fake_embeddings), samples_per_class, replace=False)
        fake_sample = fake_embeddings[fake_idx]
    else:
        fake_sample = fake_embeddings
    
    # Combine and project
    if len(real_sample) > 0 and len(fake_sample) > 0:
        all_embeddings = np.vstack([real_sample, fake_sample])
    elif len(real_sample) > 0:
        all_embeddings = real_sample
    elif len(fake_sample) > 0:
        all_embeddings = fake_sample
    else:
        raise ValueError("No embeddings found to fit PCA")
    
    # Project embeddings
    print(f"   Projecting {len(all_embeddings):,} embeddings...")
    z_auth, _ = run_model_inference(model, all_embeddings, batch_size=256, device=device)
    
    # Fit PCA
    print(f"   Fitting PCA on {len(z_auth):,} projected embeddings...")
    pca = PCA(n_components=50)
    pca.fit(z_auth)
    
    return pca


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations for experiment candidates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate visuals for hubert baseline embeddings
  python scripts/generate_experiment_visuals.py --embedding hubert --projection baseline
  
  # Generate visuals for hubert trained projections (conservative, moderate, aggressive)
  python scripts/generate_experiment_visuals.py --embedding hubert --projection conservative --pipeline-dir results/hubert_pipeline_sweep_refactor
  python scripts/generate_experiment_visuals.py --embedding hubert --projection moderate --pipeline-dir results/hubert_pipeline_sweep_refactor
  python scripts/generate_experiment_visuals.py --embedding hubert --projection aggressive --pipeline-dir results/hubert_pipeline_sweep_refactor
  
  # Generate all projections at once
  python scripts/generate_experiment_visuals.py --embedding hubert --projection all --pipeline-dir results/hubert_pipeline_sweep_refactor
        """
    )
    parser.add_argument(
        '--embedding',
        type=str,
        required=True,
        choices=['hubert', 'openl3', 'senet'],
        help='Embedding type to generate visuals for'
    )
    parser.add_argument(
        '--projection',
        type=str,
        required=True,
        choices=['baseline', 'conservative', 'moderate', 'aggressive', 'all'],
        help='Projection type (or "all" for conservative/moderate/aggressive)'
    )
    parser.add_argument(
        '--pipeline-dir',
        type=str,
        help='Path to pipeline directory containing trained models (required for non-baseline projections)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='exports/deepfake_embeddings.h5',
        help='Path to HDF5 data file (default: exports/deepfake_embeddings.h5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/final_results',
        help='Base output directory (default: results/final_results)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use for inference (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    output_base_dir = Path(args.output_dir)
    
    # Determine which projections to process
    if args.projection == 'all':
        if not args.pipeline_dir:
            print("❌ Error: --pipeline-dir required when --projection=all")
            return
        projection_types = ['conservative', 'moderate', 'aggressive']
    else:
        projection_types = [args.projection]
    
    print("=" * 80)
    print("Experiment Visualization Generator")
    print("=" * 80)
    print(f"Data file: {args.data_path}")
    print(f"Embedding type: {args.embedding}")
    print(f"Projection type(s): {', '.join(projection_types)}")
    if args.pipeline_dir:
        print(f"Pipeline directory: {args.pipeline_dir}")
    print(f"Device: {device}")
    print(f"Output base directory: {output_base_dir}")
    print(f"Number of candidates: {len(EXPERIMENT_CANDIDATES)}")
    print()
    
    # Generate visuals for each projection type
    for projection_type in projection_types:
        print("\n" + "=" * 80)
        print(f"Processing {args.embedding.upper()}/{projection_type.upper()}")
        print("=" * 80)
        print()
        
        try:
            if projection_type == 'baseline':
                # Baseline: use analyzer directly
                output_dir = output_base_dir / 'baseline' / args.embedding / 'figures'
                generate_visuals_baseline(
                    embedding_type=args.embedding,
                    data_path=args.data_path,
                    output_dir=output_dir
                )
            else:
                # Trained projections: load model and run inference
                if not args.pipeline_dir:
                    print(f"❌ Error: --pipeline-dir required for {projection_type} projection")
                    continue
                
                model_path = Path(args.pipeline_dir) / projection_type / 'best_model.pt'
                if not model_path.exists():
                    print(f"❌ Error: Model not found: {model_path}")
                    continue
                
                output_dir = output_base_dir / projection_type / args.embedding / 'figures'
                generate_visuals_projected(
                    embedding_type=args.embedding,
                    projection_type=projection_type,
                    data_path=args.data_path,
                    model_path=str(model_path),
                    output_dir=output_dir,
                    device=device
                )
        except Exception as e:
            print(f"\n❌ Error processing {args.embedding}/{projection_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("✅ All visualizations complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_base_dir}/")


if __name__ == "__main__":
    main()

