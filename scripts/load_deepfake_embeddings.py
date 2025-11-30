#!/usr/bin/env python3
"""
Load deepfake embedding data from Neon PostgreSQL and save as HDF5.

This script:
1. Queries segments and embeddings from Neon Postgres
   - AVDeepfake1M: filtered by created_at >= filter (default: '2025-11-01 00:00:00')
   - ShareVeo3: filtered by source = 'ShareVeo3' (optional, use --include-shareveo3)
   - Sora2: filtered by source = 'Sora2' (optional, use --include-sora2)
2. Groups data hierarchically by video_id and augmentations
3. Identifies source videos and classifies augmentation types
4. Saves structured data as HDF5 for efficient storage and partial loading

Usage:
    # Load only AVDeepfake1M
    python scripts/load_deepfake_embeddings.py --created-at-filter '2025-11-01 00:00:00'
    
    # Load both AVDeepfake1M and ShareVeo3
    python scripts/load_deepfake_embeddings.py --created-at-filter '2025-11-01 00:00:00' --include-shareveo3
    
    # Load Sora2 dataset
    python scripts/load_deepfake_embeddings.py --include-sora2 --output exports/sora2_embeddings.h5
"""

import argparse
import os
import sys
from datetime import datetime

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.neon import (
    connect_neon,
    get_all_video_ids,
    analyze_video_id_distribution,
    fetch_video_data,
    process_video_data,
    save_to_hdf5,
    load_from_hdf5,
    check_video_completeness,
    validate_data,
)

# Configuration
EMBEDDING_TABLES = {
    "openl3": "embeddings_audio_openl3",
    "hubert": "embeddings_audio_hubert",
    "senet": "embeddings_video_senet",
}

# Default filter date
DEFAULT_CREATED_AT_FILTER = "2025-11-01 00:00:00"


def main():
    parser = argparse.ArgumentParser(
        description="Load deepfake embeddings from Neon and save as HDF5"
    )
    parser.add_argument(
        "--created-at-filter",
        type=str,
        default=DEFAULT_CREATED_AT_FILTER,
        help=f"Filter segments by created_at >= this date (default: {DEFAULT_CREATED_AT_FILTER})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exports/deepfake_embeddings.h5",
        help="Output HDF5 filename (default: deepfake_embeddings.h5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test queries without loading all data (just count videos)",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Only load videos with all 3 embedding types complete (skip incomplete batches)",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Don't deduplicate by segment_id (include all batches even if segments overlap)",
    )
    parser.add_argument(
        "--include-shareveo3",
        action="store_true",
        help="Also load ShareVeo3 dataset (filtered by source='ShareVeo3')",
    )
    parser.add_argument(
        "--include-sora2",
        action="store_true",
        help="Also load Sora2 dataset (filtered by source='Sora2')",
    )
    parser.add_argument(
        "--exclude-incomplete-augmentations",
        action="store_true",
        help="Exclude entire augmentations if any segment is missing embeddings (default: only exclude incomplete segments)",
    )
    args = parser.parse_args()
    
    print(f"📊 Loading deepfake embeddings from Neon")
    if not args.include_sora2:
        print(f"   AVDeepfake1M filter: created_at >= {args.created_at_filter}")
    if args.include_shareveo3:
        print(f"   ShareVeo3 filter: source = 'ShareVeo3'")
    if args.include_sora2:
        print(f"   Sora2 filter: source = 'Sora2'")
    print(f"   Output: {args.output}")
    print(f"   Embedding filter: Only segments with all 3 embeddings (openl3, hubert, senet)")
    if args.exclude_incomplete_augmentations:
        print(f"   Augmentation filter: Excluding entire augmentations if any segment missing embeddings")
    else:
        print(f"   Augmentation filter: Including augmentations, excluding only incomplete segments")
    if args.require_complete:
        print(f"   Mode: Only loading videos with complete embeddings (all 3 types)")
    else:
        print(f"   Mode: Loading all videos (including incomplete embeddings)")
    if args.no_deduplicate:
        print(f"   Deduplication: DISABLED (will include all batches, even if segments overlap)")
    else:
        print(f"   Deduplication: ENABLED (will deduplicate by segment_id, keeping first occurrence)")
    
    # Connect to database
    print(f"\n🔗 Connecting to Neon...")
    conn = connect_neon()
    
    try:
        # Get all video_ids for AVDeepfake1M (only if not loading Sora2 exclusively)
        avd_video_ids = []
        if not args.include_sora2:
            # Only include videos that have segments with all 3 embeddings
            # Filter: source = 'AVDeepfake1M' AND created_at >= filter_date
            print(f"\n📥 Fetching AVDeepfake1M video_ids (with all 3 embeddings)...")
            print(f"   Filter: source = 'AVDeepfake1M' AND created_at >= {args.created_at_filter}")
            avd_video_ids = get_all_video_ids(
                conn, 
                created_at_filter=args.created_at_filter,
                source="AVDeepfake1M",
                require_all_embeddings=True
            )
            print(f"   Found {len(avd_video_ids)} unique video_ids with all embeddings")
        
        # Get all video_ids for ShareVeo3 (if requested)
        sv3_video_ids = []
        if args.include_shareveo3:
            print(f"\n📥 Fetching ShareVeo3 video_ids (with all 3 embeddings)...")
            sv3_video_ids = get_all_video_ids(
                conn, 
                source="ShareVeo3",
                require_all_embeddings=True
            )
            print(f"   Found {len(sv3_video_ids)} unique video_ids with all embeddings")
        
        # Get all video_ids for Sora2 (if requested)
        sora2_video_ids = []
        if args.include_sora2:
            print(f"\n📥 Fetching Sora2 video_ids (with all 3 embeddings)...")
            sora2_video_ids = get_all_video_ids(
                conn, 
                source="Sora2",
                require_all_embeddings=True
            )
            print(f"   Found {len(sora2_video_ids)} unique video_ids with all embeddings")
        
        if args.dry_run:
            # Analyze distribution to explain any discrepancies
            if avd_video_ids:
                print(f"\n📊 Analyzing AVDeepfake1M video_id distribution...")
                analysis = analyze_video_id_distribution(conn, args.created_at_filter)
                
                print(f"\n   Per-batch counts:")
                for created_at, count in sorted(analysis["per_batch"].items()):
                    print(f"     {created_at}: {count} videos")
                
                print(f"\n   Summary:")
                print(f"     Sum of per-batch counts: {analysis['sum_per_batch']}")
                print(f"     Unique video_ids: {analysis['unique_total']}")
                if analysis['sum_per_batch'] != analysis['unique_total']:
                    diff = analysis['sum_per_batch'] - analysis['unique_total']
                    print(f"     Difference: {diff} (same video_id in multiple batches)")
                    if analysis['duplicates']:
                        print(f"\n   Videos appearing in multiple batches:")
                        for vid_id, num_batches in analysis['duplicates'][:5]:
                            print(f"     {vid_id}: appears in {num_batches} batches")
                        if len(analysis['duplicates']) > 5:
                            print(f"     ... and {len(analysis['duplicates']) - 5} more")
            
            print(f"\n✅ Dry run complete.")
            if not args.include_sora2:
                print(f"   Would process {len(avd_video_ids)} AVDeepfake1M videos")
            if args.include_shareveo3:
                print(f"   Would process {len(sv3_video_ids)} ShareVeo3 videos")
            if args.include_sora2:
                print(f"   Would process {len(sora2_video_ids)} Sora2 videos")
            return
        
        # Process AVDeepfake1M videos (only if not loading Sora2 exclusively)
        videos_data = {}
        total_segments = 0
        incomplete_videos = []
        complete_count = 0
        incomplete_count = 0
        excluded_segments_count = 0
        
        if not args.include_sora2:
            print(f"\n🔄 Processing {len(avd_video_ids)} AVDeepfake1M videos...")
            for video_id in tqdm(avd_video_ids, desc="Processing AVDeepfake1M"):
                # Fetch video data for AVDeepfake1M
                # Note: We need to filter by source='AVDeepfake1M' AND created_at filter
                # Since fetch_video_data doesn't support source filter, we'll filter after fetching
                raw_data = fetch_video_data(
                    conn, 
                    video_id, 
                    created_at_filter=args.created_at_filter,
                    deduplicate=not args.no_deduplicate
                )
                
                # Filter to only AVDeepfake1M segments (with date filter already applied)
                raw_data = [seg for seg in raw_data if seg.get('source') == 'AVDeepfake1M']
                
                # Skip if no data after filtering
                if not raw_data:
                    continue
                
                # Count segments before filtering (for statistics)
                segments_before = len(raw_data)
                
                # Check completeness for statistics
                is_complete, stats = check_video_completeness(raw_data)
                if is_complete:
                    complete_count += 1
                else:
                    incomplete_count += 1
                    if not args.require_complete:
                        incomplete_videos.append((video_id, stats))
                
                processed = process_video_data(
                    video_id, raw_data, 
                    require_complete=args.require_complete,
                    dataset="avdeepfake1m",
                    exclude_incomplete_augmentations=args.exclude_incomplete_augmentations
                )
                
                if processed is not None:
                    videos_data[video_id] = processed
                    total_segments += processed["num_segments"]
                    # Count excluded segments (approximate: segments_before - segments in processed)
                    segments_after = processed["num_segments"] * processed["num_augmentations"]
                    excluded_segments_count += max(0, segments_before - segments_after)
        
        # Process ShareVeo3 videos (if requested)
        if args.include_shareveo3:
            print(f"\n🔄 Processing {len(sv3_video_ids)} ShareVeo3 videos...")
            for video_id in tqdm(sv3_video_ids, desc="Processing ShareVeo3"):
                raw_data = fetch_video_data(
                    conn, 
                    video_id, 
                    source="ShareVeo3",
                    deduplicate=not args.no_deduplicate
                )
                
                # Count segments before filtering (for statistics)
                segments_before = len(raw_data)
                
                # Check completeness for statistics
                is_complete, stats = check_video_completeness(raw_data)
                if is_complete:
                    complete_count += 1
                else:
                    incomplete_count += 1
                    if not args.require_complete:
                        incomplete_videos.append((video_id, stats))
                
                processed = process_video_data(
                    video_id, raw_data, 
                    require_complete=args.require_complete,
                    dataset="shareveo3",
                    exclude_incomplete_augmentations=args.exclude_incomplete_augmentations
                )
                
                if processed is not None:
                    videos_data[video_id] = processed
                    total_segments += processed["num_segments"]
                    # Count excluded segments (approximate: segments_before - segments in processed)
                    segments_after = processed["num_segments"] * processed["num_augmentations"]
                    excluded_segments_count += max(0, segments_before - segments_after)
        
        # Process Sora2 videos (if requested)
        if args.include_sora2:
            print(f"\n🔄 Processing {len(sora2_video_ids)} Sora2 videos...")
            for video_id in tqdm(sora2_video_ids, desc="Processing Sora2"):
                raw_data = fetch_video_data(
                    conn, 
                    video_id, 
                    source="Sora2",
                    deduplicate=not args.no_deduplicate
                )
                
                # Count segments before filtering (for statistics)
                segments_before = len(raw_data)
                
                # Check completeness for statistics
                is_complete, stats = check_video_completeness(raw_data)
                if is_complete:
                    complete_count += 1
                else:
                    incomplete_count += 1
                    if not args.require_complete:
                        incomplete_videos.append((video_id, stats))
                
                processed = process_video_data(
                    video_id, raw_data, 
                    require_complete=args.require_complete,
                    dataset="sora2",
                    exclude_incomplete_augmentations=args.exclude_incomplete_augmentations
                )
                
                if processed is not None:
                    videos_data[video_id] = processed
                    total_segments += processed["num_segments"]
                    # Count excluded segments (approximate: segments_before - segments in processed)
                    segments_after = processed["num_segments"] * processed["num_augmentations"]
                    excluded_segments_count += max(0, segments_before - segments_after)
        
        print(f"\n📊 Processing Summary:")
        print(f"   Complete videos: {complete_count}")
        print(f"   Incomplete videos: {incomplete_count}")
        if args.require_complete:
            print(f"   Loaded: {len(videos_data)} videos (only complete ones)")
        else:
            print(f"   Loaded: {len(videos_data)} videos (including incomplete)")
        print(f"   Total segments: {total_segments}")
        if excluded_segments_count > 0:
            print(f"   Excluded segments (missing embeddings): {excluded_segments_count}")
            print(f"   Note: Only segments with all 3 embeddings (openl3, hubert, senet) were included")
        
        if incomplete_videos and not args.require_complete:
            print(f"\n⚠️  Incomplete videos (missing some embeddings):")
            for vid_id, stats in incomplete_videos[:10]:  # Show first 10
                print(f"   {vid_id}: {stats['openl3']}/{stats['total_segments']} openl3, "
                      f"{stats['hubert']}/{stats['total_segments']} hubert, "
                      f"{stats['senet']}/{stats['total_segments']} senet")
            if len(incomplete_videos) > 10:
                print(f"   ... and {len(incomplete_videos) - 10} more")
        
        # Build final data structure
        datasets_list = []
        if not args.include_sora2:
            datasets_list.append("avdeepfake1m")
        if args.include_shareveo3:
            datasets_list.append("shareveo3")
        if args.include_sora2:
            datasets_list.append("sora2")
        
        data = {
            "metadata": {
                "datasets": datasets_list,
                "avdeepfake1m_filter": args.created_at_filter if not args.include_sora2 else None,
                "created_at_filter": args.created_at_filter if not args.include_sora2 else None,  # Keep for backward compatibility
                "embedding_types": list(EMBEDDING_TABLES.keys()),
                "video_ids": sorted(videos_data.keys()),
                "total_videos": len(videos_data),
                "total_segments": total_segments,
                "total_embeddings": sum(
                    v["embeddings"]["hubert"].size for v in videos_data.values()
                ),
                "date_created": datetime.now().isoformat(),
            },
            "videos": videos_data,
            "models": {
                "pca": {"openl3": None, "hubert": None, "senet": None},
                "umap": {"openl3": None, "hubert": None, "senet": None},
                "tsne": None,
            },
            "results": {
                "timestamp_analysis": {},
                "temporal_analysis": {},
                "aggregate_stats": None,
            },
        }
        
        # Save to HDF5
        save_to_hdf5(data, args.output)
        
        # Validate
        print(f"\n🔍 Validating saved data...")
        loaded_data = load_from_hdf5(args.output)
        validate_data(loaded_data)
        
        print(f"\n✅ All done!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
