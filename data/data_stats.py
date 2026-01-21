#!/usr/bin/env python3
"""
Script to get statistics about videos and segments in an HDF5 file.
Counts total videos, real segments, and fake segments.
"""
import h5py
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

def get_data_stats(hdf5_path: str):
    """Get statistics about videos and segments."""
    
    if not Path(hdf5_path).exists():
        print(f"❌ Error: File not found: {hdf5_path}")
        sys.exit(1)
    
    print(f"📊 Analyzing: {hdf5_path}\n")
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'videos' not in f:
                print("❌ Error: 'videos' group not found in HDF5 file")
                sys.exit(1)
            
            videos_group = f['/videos']
            video_ids = list(videos_group.keys())
            total_videos = len(videos_group.keys())
            
            print(f"📹 Found {total_videos:,} unique video IDs")
            
            # Count segments
            total_segments = 0
            real_segments = 0
            fake_segments = 0
            videos_with_labels = 0
            videos_skipped = 0
            
            print(f"\n📊 Counting segments...")
            
            for video_id in tqdm(video_ids, desc="Processing videos"):
                video = videos_group[video_id]
                
                # Check if has labels
                if 'labels/audio' not in video or 'labels/video' not in video:
                    videos_skipped += 1
                    continue
                
                videos_with_labels += 1
                
                # Load labels
                audio_labels = video['labels/audio'][:]  # [num_augs, num_segs]
                video_labels = video['labels/video'][:]  # [num_augs, num_segs]
                
                num_augs, num_segs = audio_labels.shape
                
                # Count segments (one per augmentation-segment pair)
                for aug_idx in range(num_augs):
                    for seg_idx in range(num_segs):
                        total_segments += 1
                        
                        # Real if both audio and video labels are 0
                        is_real = (audio_labels[aug_idx, seg_idx] == 0 and 
                                  video_labels[aug_idx, seg_idx] == 0)
                        
                        if is_real:
                            real_segments += 1
                        else:
                            fake_segments += 1
            
            # Print results
            print(f"\n{'='*60}")
            print(f"RESULTS")
            print(f"{'='*60}")
            print(f"Total unique video IDs:     {total_videos:,}")
            print(f"Videos with labels:         {videos_with_labels:,}")
            print(f"Videos skipped:              {videos_skipped:,}")
            print(f"\nTotal segments:             {total_segments:,}")
            print(f"  Real segments:             {real_segments:,} ({100*real_segments/total_segments:.1f}%)")
            print(f"  Fake segments:             {fake_segments:,} ({100*fake_segments/total_segments:.1f}%)")
            print(f"{'='*60}\n")
            
            # Show a few example video IDs
            if total_videos > 0:
                print(f"First 5 video IDs:")
                for i, vid_id in enumerate(video_ids[:5]):
                    print(f"  {i+1}. {vid_id}")
                
                if total_videos > 5:
                    print(f"  ... and {total_videos - 5:,} more")
            
            return {
                'total_videos': total_videos,
                'videos_with_labels': videos_with_labels,
                'videos_skipped': videos_skipped,
                'total_segments': total_segments,
                'real_segments': real_segments,
                'fake_segments': fake_segments,
            }
    
    except Exception as e:
        print(f"❌ Error reading HDF5 file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        hdf5_path = 'data/evaluation_data/deepfake_embeddings_2.h5'
        print(f"No path provided, using default: {hdf5_path}\n")
    else:
        hdf5_path = sys.argv[1]
    
    get_data_stats(hdf5_path)

