"""
Analyze real/fake segment split for AVDeepfake1M and ShareVeo3 datasets.
"""

import h5py
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def analyze_dataset_split(hdf5_path: str):
    """
    Analyze the real/fake segment distribution for AVDeepfake1M and ShareVeo3 datasets.
    
    Args:
        hdf5_path: Path to the HDF5 file containing the embeddings and labels
    """
    print(f"📂 Analyzing dataset split from: {hdf5_path}\n")
    
    # Statistics per dataset
    stats = {
        'avdeepfake1m': {
            'total_videos': 0,
            'total_segments': 0,
            'real_segments': 0,
            'fake_segments': 0,
            'total_augmentations': 0
        },
        'shareveo3': {
            'total_videos': 0,
            'total_segments': 0,
            'real_segments': 0,
            'fake_segments': 0,
            'total_augmentations': 0
        }
    }
    
    try:
        with h5py.File(hdf5_path, "r") as f:
            # Get all video IDs
            if "metadata" in f and "video_ids" in f["metadata"]:
                video_ids = [vid.decode() if isinstance(vid, bytes) else vid 
                            for vid in f["metadata"]["video_ids"][:]]
            elif "videos" in f:
                video_ids = list(f["videos"].keys())
            else:
                print("❌ Could not find video IDs in HDF5 file")
                return
            
            print(f"Found {len(video_ids)} videos\n")
            print("Processing videos...")
            
            # Process each video
            for video_id in tqdm(video_ids, desc="Analyzing videos"):
                safe_video_id = video_id.replace("/", "_")
                
                try:
                    vid_grp = f["videos"][safe_video_id]
                    
                    # Get dataset source
                    dataset = vid_grp.attrs.get('dataset', 'avdeepfake1m')
                    if isinstance(dataset, bytes):
                        dataset = dataset.decode()
                    
                    # Normalize dataset name
                    if dataset.lower() in ['avdeepfake1m', 'avdeepfake', 'av1m']:
                        dataset_key = 'avdeepfake1m'
                    elif dataset.lower() in ['shareveo3', 'share_veo3', 'veo3']:
                        dataset_key = 'shareveo3'
                    else:
                        # Skip unknown datasets
                        continue
                    
                    # Load labels (using audio labels as default)
                    if "labels" not in vid_grp:
                        continue
                    
                    labels_grp = vid_grp["labels"]
                    if "audio" not in labels_grp:
                        continue
                    
                    labels = labels_grp["audio"][:]  # Shape: [num_augs, num_segs]
                    
                    # Get number of augmentations and segments
                    num_augs, num_segs = labels.shape
                    
                    # Count real/fake segments
                    # Original convention: 1=fake, 0=real
                    # We flip to standard convention: 1=real, 0=fake
                    # Labels > 0.5 (original fake) → 0 (fake), Labels <= 0.5 (original real) → 1 (real)
                    binary_labels = (labels <= 0.5).astype(int)
                    real_count = np.sum(binary_labels == 1)
                    fake_count = np.sum(binary_labels == 0)
                    
                    # Update statistics
                    stats[dataset_key]['total_videos'] += 1
                    stats[dataset_key]['total_segments'] += num_segs * num_augs
                    stats[dataset_key]['real_segments'] += real_count
                    stats[dataset_key]['fake_segments'] += fake_count
                    stats[dataset_key]['total_augmentations'] += num_augs
                    
                except Exception as e:
                    print(f"\n⚠️ Error processing video {video_id}: {e}")
                    continue
            
            # Print results
            print("\n" + "=" * 80)
            print("DATASET REAL/FAKE SEGMENT ANALYSIS")
            print("=" * 80)
            
            for dataset_name, dataset_stats in stats.items():
                print(f"\n📊 {dataset_name.upper()}")
                print("-" * 80)
                print(f"  Total Videos:           {dataset_stats['total_videos']:,}")
                print(f"  Total Augmentations:    {dataset_stats['total_augmentations']:,}")
                print(f"  Total Segments:         {dataset_stats['total_segments']:,}")
                print(f"  Real Segments:          {dataset_stats['real_segments']:,} ({dataset_stats['real_segments']/dataset_stats['total_segments']*100:.2f}%)")
                print(f"  Fake Segments:         {dataset_stats['fake_segments']:,} ({dataset_stats['fake_segments']/dataset_stats['total_segments']*100:.2f}%)")
                
                if dataset_stats['total_videos'] > 0:
                    avg_segs_per_video = dataset_stats['total_segments'] / dataset_stats['total_videos']
                    avg_augs_per_video = dataset_stats['total_augmentations'] / dataset_stats['total_videos']
                    print(f"  Avg Segments/Video:     {avg_segs_per_video:.2f}")
                    print(f"  Avg Augmentations/Video: {avg_augs_per_video:.2f}")
            
            # Overall summary
            print("\n" + "=" * 80)
            print("OVERALL SUMMARY")
            print("=" * 80)
            total_videos = sum(s['total_videos'] for s in stats.values())
            total_segments = sum(s['total_segments'] for s in stats.values())
            total_real = sum(s['real_segments'] for s in stats.values())
            total_fake = sum(s['fake_segments'] for s in stats.values())
            
            print(f"  Total Videos:           {total_videos:,}")
            print(f"  Total Segments:         {total_segments:,}")
            print(f"  Real Segments:          {total_real:,} ({total_real/total_segments*100:.2f}%)")
            print(f"  Fake Segments:         {total_fake:,} ({total_fake/total_segments*100:.2f}%)")
            print(f"  Real/Fake Ratio:       {total_real/total_fake:.3f}")
            
    except FileNotFoundError:
        print(f"❌ File not found: {hdf5_path}")
        print("Please check the path and try again.")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Try to find the HDF5 file
    import os
    
    # Check common locations
    possible_paths = [
        "./deepfake_embeddings_2.h5",
        "/Users/jerrysheng/Downloads/itau-group4/deepfake_embeddings_2.h5",
        "/Users/jerrysheng/Desktop/Lab/deepfake_embeddings_2.h5",
    ]
    
    hdf5_path = None
    for path in possible_paths:
        if os.path.exists(path):
            hdf5_path = path
            break
    
    if hdf5_path is None:
        print("❌ Could not find deepfake_embeddings_2.h5 file")
        print("Please specify the path to the HDF5 file:")
        print("  python analyze_dataset_split.py <path_to_hdf5_file>")
        import sys
        if len(sys.argv) > 1:
            hdf5_path = sys.argv[1]
        else:
            sys.exit(1)
    
    analyze_dataset_split(hdf5_path)

