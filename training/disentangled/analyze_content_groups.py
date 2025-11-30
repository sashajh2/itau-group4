"""
Analyze content group distribution in the dataset.
Helps determine minimum samples per content group for training.
"""
import argparse
import h5py
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path


def analyze_content_groups(hdf5_path: str, encoder_name: str = 'hubert', output_path: str = None):
    """
    Analyze the distribution of samples per content group.
    
    Args:
        hdf5_path: Path to HDF5 file
        encoder_name: Encoder name to check
        output_path: Optional path to save analysis CSV
    """
    print(f"📊 Analyzing content group distribution...")
    print(f"   HDF5: {hdf5_path}")
    print(f"   Encoder: {encoder_name}\n")
    
    content_group_counts = Counter()
    real_counts = Counter()
    fake_counts = Counter()
    
    total_videos = 0
    total_samples = 0
    
    with h5py.File(hdf5_path, 'r') as f:
        videos_group = f['/videos']
        video_ids = list(videos_group.keys())
        total_videos = len(video_ids)
        
        for video_idx, video_id in enumerate(video_ids):
            if (video_idx + 1) % 1000 == 0:
                print(f"   Processed {video_idx + 1}/{total_videos} videos...")
            
            video = videos_group[video_id]
            
            if 'augmentation_info' not in video:
                continue
            
            source_idx = int(video['augmentation_info'].attrs.get('source_idx', 0))
            
            if f'embeddings/{encoder_name}' not in video:
                continue
            
            embeddings = video[f'embeddings/{encoder_name}'][:]  # [num_augs, num_segs, emb_dim]
            audio_labels = video['labels/audio'][:]  # [num_augs, num_segs]
            video_labels = video['labels/video'][:]  # [num_augs, num_segs]
            
            num_augs, num_segs, emb_dim = embeddings.shape
            
            # Count samples per content group
            for aug_idx in range(num_augs):
                for seg_idx in range(num_segs):
                    content_group = (source_idx, seg_idx)
                    content_group_counts[content_group] += 1
                    total_samples += 1
                    
                    is_real = (audio_labels[aug_idx, seg_idx] == 0) and (video_labels[aug_idx, seg_idx] == 0)
                    if is_real:
                        real_counts[content_group] += 1
                    else:
                        fake_counts[content_group] += 1
    
    print(f"\n✅ Analysis complete!")
    print(f"   Total videos: {total_videos:,}")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Unique content groups: {len(content_group_counts):,}\n")
    
    # Create DataFrame for analysis
    group_sizes = list(content_group_counts.values())
    real_per_group = [real_counts.get(cg, 0) for cg in content_group_counts.keys()]
    fake_per_group = [fake_counts.get(cg, 0) for cg in content_group_counts.keys()]
    
    df = pd.DataFrame({
        'content_group': list(content_group_counts.keys()),
        'total_samples': group_sizes,
        'real_samples': real_per_group,
        'fake_samples': fake_per_group,
    })
    
    # Print statistics
    print("=" * 60)
    print("Content Group Size Statistics")
    print("=" * 60)
    print(df['total_samples'].describe())
    print()
    
    print("Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(group_sizes, p)
        count = sum(1 for s in group_sizes if s >= val)
        print(f"  {p}th percentile: {val:.1f} samples ({count:,} groups have ≥{val:.0f} samples)")
    print()
    
    # Show distribution
    print("Distribution by size:")
    size_bins = [1, 2, 3, 4, 5, 10, 20, 50, 100, float('inf')]
    for i in range(len(size_bins) - 1):
        min_size = size_bins[i]
        max_size = size_bins[i + 1]
        if max_size == float('inf'):
            count = sum(1 for s in group_sizes if s >= min_size)
            print(f"  ≥{min_size} samples: {count:,} groups")
        else:
            count = sum(1 for s in group_sizes if min_size <= s < max_size)
            print(f"  {min_size}-{max_size-1} samples: {count:,} groups")
    print()
    
    # Real/Fake distribution
    print("Real/Fake distribution per group:")
    print(f"  Groups with only real samples: {sum(1 for r, f in zip(real_per_group, fake_per_group) if f == 0):,}")
    print(f"  Groups with only fake samples: {sum(1 for r, f in zip(real_per_group, fake_per_group) if r == 0):,}")
    print(f"  Groups with both: {sum(1 for r, f in zip(real_per_group, fake_per_group) if r > 0 and f > 0):,}")
    print()
    
    # Recommendations
    print("=" * 60)
    print("Recommendations")
    print("=" * 60)
    
    # Find minimum size that covers most groups
    for min_size in [2, 3, 4, 5, 10]:
        groups_above = sum(1 for s in group_sizes if s >= min_size)
        samples_above = sum(s for s in group_sizes if s >= min_size)
        pct_groups = 100 * groups_above / len(group_sizes) if len(group_sizes) > 0 else 0
        pct_samples = 100 * samples_above / total_samples if total_samples > 0 else 0
        print(f"  min_samples={min_size}: {groups_above:,} groups ({pct_groups:.1f}%), "
              f"{samples_above:,} samples ({pct_samples:.1f}%)")
    
    print()
    print("💡 Recommendation: Use min_samples=3 or 4 for stable prototypical learning")
    print("   (Need at least 2 samples per group for prototype computation)")
    print("=" * 60)
    
    # Save to CSV if requested
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n💾 Saved analysis to {output_path}")
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze content group distribution")
    parser.add_argument('--hdf5-path', type=str, required=True,
                       help='Path to HDF5 file')
    parser.add_argument('--encoder-name', type=str, default='hubert',
                       choices=['hubert', 'openl3', 'senet'],
                       help='Encoder name')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional path to save CSV analysis')
    
    args = parser.parse_args()
    analyze_content_groups(args.hdf5_path, args.encoder_name, args.output)

