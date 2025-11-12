#!/usr/bin/env python3
"""
Run baseline embedding analysis experiments on selected video/timestamp combinations.

This script:
1. Loads the HDF5 data file
2. Fits global PCA
3. Runs Experiment 1 (cross-augmentation analysis) on multiple video/timestamp pairs
4. Runs Experiment 3 (aggregate analysis)
5. Saves all results and figures

Supports multiple embedding types: hubert, openl3, senet
Results are organized by embedding type in subdirectories.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.analyzer import DeepfakeEmbeddingAnalyzer


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


def run_experiments_for_embedding_type(embedding_type: str, data_path: str, output_base_dir: Path):
    """
    Run all experiments for a specific embedding type.
    
    Args:
        embedding_type: 'hubert', 'openl3', or 'senet'
        data_path: Path to HDF5 data file
        output_base_dir: Base output directory (results will be in subdirectories)
    """
    # Create embedding-specific output directories
    output_dir = output_base_dir / embedding_type
    figures_dir = output_dir / 'figures'
    metrics_dir = output_dir / 'metrics'
    
    # Create output directories
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Baseline Embedding Analysis Experiments: {embedding_type.upper()}")
    print("=" * 80)
    print(f"Data file: {data_path}")
    print(f"Embedding type: {embedding_type}")
    print(f"Output directory: {output_dir}")
    print(f"Number of experiments: {len(EXPERIMENT_CANDIDATES)}")
    print()
    
    # Initialize analyzer
    print("📊 Initializing analyzer...")
    analyzer = DeepfakeEmbeddingAnalyzer(data_path, embedding_type=embedding_type)
    print(f"   Loaded metadata: {analyzer.data['total_videos']} videos")
    print()
    
    # Fit PCA (one-time, used for all visualizations)
    print("🔧 Fitting global PCA...")
    analyzer.fit_global_pca(n_components=50, sample_size=50000)
    print()
    
    # Create global PCA visualization
    print("=" * 80)
    print("Global PCA Visualization")
    print("=" * 80)
    global_pca_fig = figures_dir / f'global_pca_all_embeddings_{embedding_type}.png'
    analyzer.plot_global_pca(sample_size=20000, save_fig=str(global_pca_fig))
    print()
    
    # Run Experiment 1: Cross-augmentation analysis for each candidate
    print("=" * 80)
    print("Experiment 1: Cross-Augmentation Analysis")
    print("=" * 80)
    
    experiment1_results = []
    
    for idx, (video_id, segment_index, num_fake, num_real, avg_label) in enumerate(EXPERIMENT_CANDIDATES, 1):
        print(f"\n[{idx}/{len(EXPERIMENT_CANDIDATES)}] Analyzing {video_id} @ segment {segment_index}")
        print(f"   Expected: {num_fake} fake, {num_real} real augmentations, avg_label={avg_label:.3f}")
        
        # Create safe filename with embedding type
        safe_video_id = video_id.replace('/', '_')
        fig_path = figures_dir / f'exp1_{embedding_type}_{safe_video_id}_seg{segment_index}.png'
        
        try:
            metrics = analyzer.experiment1_cross_aug_single_timestamp(
                video_id=video_id,
                timestamp_idx=segment_index,
                save_fig=str(fig_path)
            )
            
            if metrics is not None:
                # Add metadata
                metrics['num_fake_expected'] = num_fake
                metrics['num_real_expected'] = num_real
                metrics['avg_label_expected'] = avg_label
                experiment1_results.append(metrics)
                print(f"   ✓ Saved figure: {fig_path.name}")
                print(f"   Silhouette score: {metrics.get('silhouette_score', 'N/A')}")
            else:
                print(f"   ⚠️  Skipped (single augmentation video)")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Save Experiment 1 results
    if experiment1_results:
        df_exp1 = pd.DataFrame(experiment1_results)
        exp1_csv = metrics_dir / f'experiment1_cross_augmentation_{embedding_type}.csv'
        df_exp1.to_csv(exp1_csv, index=False)
        print(f"\n✅ Saved Experiment 1 results: {exp1_csv}")
        print(f"   Total experiments: {len(df_exp1)}")
        print(f"\n   Summary statistics:")
        print(df_exp1[['silhouette_score', 'centroid_cosine_distance']].describe())
    
    # Run Experiment 3: Aggregate analysis
    print("\n" + "=" * 80)
    print("Experiment 3: Aggregate Analysis")
    print("=" * 80)
    
    print("\n📊 Running aggregate analysis across all videos...")
    df_exp3 = analyzer.experiment3_aggregate_analysis(max_videos=None)  # Analyze all videos
    
    if len(df_exp3) > 0:
        exp3_csv = metrics_dir / f'experiment3_aggregate_analysis_{embedding_type}.csv'
        df_exp3.to_csv(exp3_csv, index=False)
        print(f"✅ Saved Experiment 3 results: {exp3_csv}")
        print(f"   Total (video_id, timestamp_idx) pairs: {len(df_exp3)}")
        print(f"\n   Summary statistics:")
        print(df_exp3[['silhouette_score', 'centroid_cosine_distance']].describe())
        
        # Additional analysis: best and worst timestamps
        print(f"\n   Top 5 timestamps by silhouette score:")
        top5 = df_exp3.nlargest(5, 'silhouette_score')[['video_id', 'timestamp_idx', 'silhouette_score', 'centroid_cosine_distance']]
        print(top5.to_string(index=False))
        
        print(f"\n   Bottom 5 timestamps by silhouette score:")
        bottom5 = df_exp3.nsmallest(5, 'silhouette_score')[['video_id', 'timestamp_idx', 'silhouette_score', 'centroid_cosine_distance']]
        print(bottom5.to_string(index=False))
    else:
        print("⚠️  No results from aggregate analysis")
    
    # Compute aggregate statistics
    print("\n" + "=" * 80)
    print("Aggregate Statistics: Real vs Fake Similarity to Source")
    print("=" * 80)
    
    agg_stats = analyzer.compute_aggregate_statistics(max_videos=None)
    
    print(f"\n📊 Results across {agg_stats['num_videos_analyzed']} AVDeepfake1M videos:")
    print(f"\n   Real augmentations to source:")
    print(f"     Mean cosine similarity: {agg_stats['mean_cos_sim_real_to_source']:.4f}")
    print(f"     Std cosine similarity:  {agg_stats['std_cos_sim_real_to_source']:.4f}")
    print(f"     Number of samples:      {agg_stats['num_real_samples']:,}")
    
    print(f"\n   Fake augmentations to source:")
    print(f"     Mean cosine similarity: {agg_stats['mean_cos_sim_fake_to_source']:.4f}")
    print(f"     Std cosine similarity:  {agg_stats['std_cos_sim_fake_to_source']:.4f}")
    print(f"     Number of samples:      {agg_stats['num_fake_samples']:,}")
    
    if agg_stats['difference_real_vs_fake'] is not None:
        diff = agg_stats['difference_real_vs_fake']
        print(f"\n   Difference (Real - Fake): {diff:+.4f}")
        if diff > 0:
            print(f"     → Real augmentations are {diff:.4f} MORE similar to source")
        else:
            print(f"     → Fake augmentations are {abs(diff):.4f} MORE similar to source")
    
    print(f"\n   Overall linear separability (label 0 vs label != 0):")
    if agg_stats['overall_linear_separability'] is not None:
        print(f"     Cross-validated accuracy: {agg_stats['overall_linear_separability']:.4f}")
        print(f"     (Note: This is classification accuracy, not AUC)")
    else:
        print(f"     Could not compute (insufficient data)")
    
    # Save aggregate statistics
    agg_stats_json = metrics_dir / f'aggregate_statistics_{embedding_type}.json'
    # Convert numpy types to native Python types for JSON
    agg_stats_serializable = {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in agg_stats.items()
    }
    with open(agg_stats_json, 'w') as f:
        json.dump(agg_stats_serializable, f, indent=2)
    print(f"\n✅ Saved aggregate statistics: {agg_stats_json}")
    
    print("\n" + "=" * 80)
    print(f"✅ All experiments complete for {embedding_type.upper()}!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  Figures: {figures_dir}")
    print(f"  Metrics: {metrics_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Run baseline embedding analysis experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for all embedding types
  python scripts/run_baseline_experiments.py --all
  
  # Run for specific embedding type
  python scripts/run_baseline_experiments.py --embedding hubert
  python scripts/run_baseline_experiments.py --embedding openl3
  python scripts/run_baseline_experiments.py --embedding senet
        """
    )
    parser.add_argument(
        '--embedding', 
        type=str, 
        choices=['hubert', 'openl3', 'senet'],
        help='Embedding type to analyze (default: hubert)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run experiments for all embedding types (hubert, openl3, senet)'
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
        default='results/baseline',
        help='Base output directory (default: results/baseline)'
    )
    
    args = parser.parse_args()
    
    # Determine which embedding types to run
    if args.all:
        embedding_types = ['hubert', 'openl3', 'senet']
    elif args.embedding:
        embedding_types = [args.embedding]
    else:
        # Default to hubert
        embedding_types = ['hubert']
    
    output_base_dir = Path(args.output_dir)
    
    print("=" * 80)
    print("Baseline Embedding Analysis Experiments")
    print("=" * 80)
    print(f"Data file: {args.data_path}")
    print(f"Embedding types: {', '.join(embedding_types)}")
    print(f"Output base directory: {output_base_dir}")
    print()
    
    # Run experiments for each embedding type
    for embedding_type in embedding_types:
        print("\n" + "=" * 80)
        print(f"Starting experiments for {embedding_type.upper()}")
        print("=" * 80)
        print()
        
        try:
            run_experiments_for_embedding_type(
                embedding_type=embedding_type,
                data_path=args.data_path,
                output_base_dir=output_base_dir
            )
        except Exception as e:
            print(f"\n❌ Error running experiments for {embedding_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("✅ All experiments complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_base_dir}/")
    for embedding_type in embedding_types:
        print(f"  {embedding_type}/")
        print(f"    figures/")
        print(f"    metrics/")


if __name__ == "__main__":
    main()

