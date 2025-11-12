#!/usr/bin/env python3
"""
Run baseline embedding analysis experiments on selected video/timestamp combinations.

This script:
1. Loads the HDF5 data file
2. Fits global PCA
3. Runs Experiment 1 (cross-augmentation analysis) on multiple video/timestamp pairs
4. Runs Experiment 3 (aggregate analysis)
5. Saves all results and figures
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.analyzer import DeepfakeEmbeddingAnalyzer


# Query 3 results - top 20 candidates
# Format: (video_id, segment_index, num_fake, num_real, avg_label)
EXPERIMENT_CANDIDATES = [
    ('gqpErbFnbiY/00017', 4, 15, 5, 0.750),
    ('gqpErbFnbiY/00017', 5, 15, 5, 0.750),
    ('gqpErbFnbiY/00017', 6, 15, 5, 0.750),
    ('gqpErbFnbiY/00017', 7, 15, 5, 0.750),
    ('gqpErbFnbiY/00015', 4, 15, 5, 0.750),
    ('gqpErbFnbiY/00016', 4, 10, 10, 0.500),
    ('gdg4mUSwuhl/00002', 4, 10, 10, 0.500),
    ('gqpErbFnbiY/00002', 4, 10, 10, 0.500),
    ('gqpErbFnbiY/00015', 5, 10, 10, 0.500),
    ('golS4kh8ETY/00002', 4, 10, 10, 0.500),
    ('gqpErbFnbiY/00026', 4, 10, 10, 0.500),
    ('gjCwsdCssdk/00019', 4, 10, 10, 0.500),
    ('gqpErbFnbiY/00022', 4, 10, 10, 0.500),
    ('gqpErbFnbiY/00018', 4, 10, 10, 0.500),
    ('gqpErbFnbiY/00010', 4, 10, 10, 0.500),
    ('gqpErbFnbiY/00016', 5, 10, 10, 0.500),
    ('gqpErbFnbiY/00025', 4, 10, 9, 0.526),
    ('gqpErbFnbiY/00007', 4, 10, 9, 0.526),
    ('gqpErbFnbiY/00008', 4, 10, 9, 0.526),
    ('gqpErbFnbiY/00017', 8, 15, 5, 0.750),
]


def main():
    # Configuration
    data_path = 'exports/deepfake_embeddings.h5'
    embedding_type = 'hubert'
    output_dir = Path('results/baseline')
    figures_dir = output_dir / 'figures'
    metrics_dir = output_dir / 'metrics'
    
    # Create output directories
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Baseline Embedding Analysis Experiments")
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
    
    # Run Experiment 1: Cross-augmentation analysis for each candidate
    print("=" * 80)
    print("Experiment 1: Cross-Augmentation Analysis")
    print("=" * 80)
    
    experiment1_results = []
    
    for idx, (video_id, segment_index, num_fake, num_real, avg_label) in enumerate(EXPERIMENT_CANDIDATES, 1):
        print(f"\n[{idx}/{len(EXPERIMENT_CANDIDATES)}] Analyzing {video_id} @ segment {segment_index}")
        print(f"   Expected: {num_fake} fake, {num_real} real augmentations, avg_label={avg_label:.3f}")
        
        # Create safe filename
        safe_video_id = video_id.replace('/', '_')
        fig_path = figures_dir / f'exp1_{safe_video_id}_seg{segment_index}.png'
        
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
        exp1_csv = metrics_dir / 'experiment1_cross_augmentation.csv'
        df_exp1.to_csv(exp1_csv, index=False)
        print(f"\n✅ Saved Experiment 1 results: {exp1_csv}")
        print(f"   Total experiments: {len(df_exp1)}")
        print(f"\n   Summary statistics:")
        print(df_exp1[['silhouette_score', 'centroid_cosine_distance', 'linear_separability']].describe())
    
    # Run Experiment 3: Aggregate analysis
    print("\n" + "=" * 80)
    print("Experiment 3: Aggregate Analysis")
    print("=" * 80)
    
    print("\n📊 Running aggregate analysis across all videos...")
    df_exp3 = analyzer.experiment3_aggregate_analysis(max_videos=None)  # Analyze all videos
    
    if len(df_exp3) > 0:
        exp3_csv = metrics_dir / 'experiment3_aggregate_analysis.csv'
        df_exp3.to_csv(exp3_csv, index=False)
        print(f"✅ Saved Experiment 3 results: {exp3_csv}")
        print(f"   Total (video_id, timestamp_idx) pairs: {len(df_exp3)}")
        print(f"\n   Summary statistics:")
        print(df_exp3[['silhouette_score', 'centroid_cosine_distance', 'linear_separability']].describe())
        
        # Additional analysis: best and worst timestamps
        print(f"\n   Top 5 timestamps by silhouette score:")
        top5 = df_exp3.nlargest(5, 'silhouette_score')[['video_id', 'timestamp_idx', 'silhouette_score', 'centroid_cosine_distance']]
        print(top5.to_string(index=False))
        
        print(f"\n   Bottom 5 timestamps by silhouette score:")
        bottom5 = df_exp3.nsmallest(5, 'silhouette_score')[['video_id', 'timestamp_idx', 'silhouette_score', 'centroid_cosine_distance']]
        print(bottom5.to_string(index=False))
    else:
        print("⚠️  No results from aggregate analysis")
    
    print("\n" + "=" * 80)
    print("✅ All experiments complete!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  Figures: {figures_dir}")
    print(f"  Metrics: {metrics_dir}")


if __name__ == "__main__":
    main()

