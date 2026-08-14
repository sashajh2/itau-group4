"""
Analyze augmentation structure in the HDF5 embeddings file.
Shows distribution of real/fake augmentations per video and filtering options.
"""
import h5py
import numpy as np
from collections import defaultdict


def analyze_augmentations(hdf5_path='exports/deepfake_embeddings.h5', min_real=5, min_fake=5):
    """
    Analyze augmentation distribution and filter videos.

    Args:
        hdf5_path: Path to HDF5 embeddings file
        min_real: Minimum number of real augmentations (source + real)
        min_fake: Minimum number of fake augmentations

    Returns:
        List of filtered video info dicts
    """
    with h5py.File(hdf5_path, 'r') as f:
        videos = f['videos']

        video_aug_counts = []

        for vid_id in videos.keys():
            vid = videos[vid_id]

            if 'augmentation_info' not in vid or 'types' not in vid['augmentation_info']:
                continue

            aug_types = vid['augmentation_info']['types'][:]
            aug_types = [t.decode() if isinstance(t, bytes) else t for t in aug_types]

            real_count = sum(1 for t in aug_types if t in ['source', 'real'])
            fake_count = sum(1 for t in aug_types if t == 'fake')

            video_aug_counts.append({
                'vid_id': vid_id,
                'real_count': real_count,
                'fake_count': fake_count,
                'total': len(aug_types)
            })

        # Print distribution
        print('=' * 60)
        print('AUGMENTATION COUNT DISTRIBUTION')
        print('=' * 60)
        print(f'Total videos: {len(video_aug_counts)}')

        print(f'\nReal augmentations per video:')
        real_counts = [v['real_count'] for v in video_aug_counts]
        for count in sorted(set(real_counts)):
            num_videos = sum(1 for r in real_counts if r == count)
            print(f'  {count} real augs: {num_videos} videos')

        print(f'\nFake augmentations per video:')
        fake_counts = [v['fake_count'] for v in video_aug_counts]
        for count in sorted(set(fake_counts)):
            num_videos = sum(1 for f in fake_counts if f == count)
            print(f'  {count} fake augs: {num_videos} videos')

        # Filter
        filtered = [v for v in video_aug_counts
                   if v['real_count'] >= min_real and v['fake_count'] >= min_fake]

        print(f'\n{"=" * 60}')
        print(f'AFTER FILTERING (>= {min_real} real AND >= {min_fake} fake)')
        print('=' * 60)
        print(f'Videos remaining: {len(filtered)} / {len(video_aug_counts)}')

        if filtered:
            total_segments = 0
            for v in filtered:
                vid = videos[v['vid_id']]
                num_segs = vid['embeddings/hubert'].shape[1]
                total_segments += (min_real + min_fake) * num_segs

            print(f'Total samples ({min_real}+{min_fake} augs × segments): {total_segments:,}')

            print(f'\nSample filtered videos:')
            for v in filtered[:10]:
                print(f'  {v["vid_id"]}: {v["real_count"]} real, {v["fake_count"]} fake')

        return filtered


def get_filtered_video_ids(hdf5_path='exports/deepfake_embeddings.h5', min_real=5, min_fake=5):
    """
    Get list of video IDs that pass the filter.

    Returns:
        List of video_id strings
    """
    filtered = analyze_augmentations(hdf5_path, min_real, min_fake)
    return [v['vid_id'] for v in filtered]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze augmentation structure')
    parser.add_argument('--hdf5', default='exports/deepfake_embeddings.h5', help='Path to HDF5 file')
    parser.add_argument('--min-real', type=int, default=5, help='Minimum real augmentations')
    parser.add_argument('--min-fake', type=int, default=5, help='Minimum fake augmentations')

    args = parser.parse_args()

    analyze_augmentations(args.hdf5, args.min_real, args.min_fake)
