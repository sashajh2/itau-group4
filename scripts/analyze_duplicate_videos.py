#!/usr/bin/env python3
"""
Analyze duplicate video_ids across created_at batches.

This script investigates:
1. How many segments per video_id in each batch
2. Whether segments are exact duplicates
3. Whether segments differ (different augmentations, etc.)
4. Recommendations for handling duplicates

Usage:
    python scripts/analyze_duplicate_videos.py [--created-at-filter '2025-11-01 00:00:00']
"""

import argparse
import sys
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import psycopg2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config

DEFAULT_CREATED_AT_FILTER = "2025-11-01 00:00:00"


def connect_neon():
    """Connect to Neon Postgres database."""
    cfg = load_config()
    dsn = cfg["database"]["postgres"]["neon_database_url"]
    return psycopg2.connect(dsn)


def find_duplicate_video_ids(conn: psycopg2.extensions.connection, created_at_filter: str) -> List[Tuple[str, int]]:
    """Find video_ids that appear in multiple batches."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                video_id,
                COUNT(DISTINCT created_at) as num_batches
            FROM segments
            WHERE created_at >= %s
            GROUP BY video_id
            HAVING COUNT(DISTINCT created_at) > 1
            ORDER BY num_batches DESC, video_id
        """, (created_at_filter,))
        return cur.fetchall()


def analyze_video_id_across_batches(
    conn: psycopg2.extensions.connection, video_id: str, created_at_filter: str
) -> Dict:
    """
    Analyze a specific video_id across all batches.
    
    Returns:
        Dictionary with detailed analysis
    """
    with conn.cursor() as cur:
        # Get all segments for this video_id
        cur.execute("""
            SELECT 
                segment_id,
                source,
                video_path,
                start_time,
                duration,
                audio_label,
                video_label,
                audio_model,
                video_model,
                created_at
            FROM segments
            WHERE video_id = %s
              AND created_at >= %s
            ORDER BY created_at, video_path, start_time
        """, (video_id, created_at_filter))
        
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        segments = [dict(zip(columns, row)) for row in rows]
    
    # Group by created_at
    by_created_at: Dict[str, List[Dict]] = defaultdict(list)
    for seg in segments:
        by_created_at[str(seg["created_at"])].append(seg)
    
    # Analyze
    analysis = {
        "video_id": video_id,
        "batches": list(by_created_at.keys()),
        "num_batches": len(by_created_at),
        "segments_per_batch": {batch: len(segs) for batch, segs in by_created_at.items()},
        "total_segments": len(segments),
        "unique_segment_ids": len(set(seg["segment_id"] for seg in segments)),
        "unique_video_paths": len(set(seg["video_path"] for seg in segments)),
        "video_paths_per_batch": {},
        "exact_duplicates": [],
        "segment_id_overlap": {},
    }
    
    # Check for exact duplicate segment_ids
    segment_id_counts = defaultdict(int)
    for seg in segments:
        segment_id_counts[seg["segment_id"]] += 1
    
    exact_duplicate_segment_ids = {
        seg_id: count for seg_id, count in segment_id_counts.items() if count > 1
    }
    analysis["exact_duplicate_segment_ids"] = exact_duplicate_segment_ids
    
    # Get video_paths per batch
    for batch, segs in by_created_at.items():
        analysis["video_paths_per_batch"][batch] = sorted(set(seg["video_path"] for seg in segs))
    
    # Check segment_id overlap between batches
    batch_names = sorted(by_created_at.keys())
    for i, batch1 in enumerate(batch_names):
        for batch2 in batch_names[i + 1 :]:
            seg_ids1 = set(seg["segment_id"] for seg in by_created_at[batch1])
            seg_ids2 = set(seg["segment_id"] for seg in by_created_at[batch2])
            overlap = seg_ids1 & seg_ids2
            analysis["segment_id_overlap"][f"{batch1} vs {batch2}"] = {
                "overlap_count": len(overlap),
                "batch1_only": len(seg_ids1 - seg_ids2),
                "batch2_only": len(seg_ids2 - seg_ids1),
                "overlap_segment_ids": sorted(list(overlap))[:10],  # First 10 for display
            }
    
    # Check for exact duplicate rows (all fields same except created_at)
    # Group by all fields except created_at
    key_to_segments = defaultdict(list)
    for seg in segments:
        # Create a key from all fields except created_at
        key = (
            seg["segment_id"],
            seg["source"],
            seg["video_path"],
            seg["start_time"],
            seg["duration"],
            seg["audio_label"],
            seg["video_label"],
            seg["audio_model"],
            seg["video_model"],
        )
        key_to_segments[key].append(seg)
    
    exact_duplicate_rows = {
        key: segs
        for key, segs in key_to_segments.items()
        if len(segs) > 1
    }
    analysis["exact_duplicate_rows"] = {
        "count": len(exact_duplicate_rows),
        "examples": list(exact_duplicate_rows.items())[:5],  # First 5 examples
    }
    
    return analysis


def print_analysis(analysis: Dict):
    """Print detailed analysis for a video_id."""
    print(f"\n{'='*80}")
    print(f"📊 Analysis for video_id: {analysis['video_id']}")
    print(f"{'='*80}")
    
    print(f"\n📦 Batch Information:")
    print(f"   Number of batches: {analysis['num_batches']}")
    print(f"   Batches: {', '.join(analysis['batches'])}")
    
    print(f"\n📈 Segment Counts:")
    print(f"   Total segments: {analysis['total_segments']}")
    print(f"   Unique segment_ids: {analysis['unique_segment_ids']}")
    print(f"   Unique video_paths: {analysis['unique_video_paths']}")
    print(f"   Segments per batch:")
    for batch, count in sorted(analysis["segments_per_batch"].items()):
        print(f"     {batch}: {count} segments")
    
    print(f"\n🎬 Video Paths per Batch:")
    for batch, paths in sorted(analysis["video_paths_per_batch"].items()):
        print(f"   {batch}:")
        for path in paths[:5]:  # Show first 5
            print(f"     - {path}")
        if len(paths) > 5:
            print(f"     ... and {len(paths) - 5} more")
    
    print(f"\n🔄 Segment ID Overlap Between Batches:")
    if analysis["segment_id_overlap"]:
        for comparison, overlap_info in analysis["segment_id_overlap"].items():
            print(f"   {comparison}:")
            print(f"     Overlap: {overlap_info['overlap_count']} segment_ids")
            print(f"     Batch 1 only: {overlap_info['batch1_only']}")
            print(f"     Batch 2 only: {overlap_info['batch2_only']}")
            if overlap_info["overlap_segment_ids"]:
                print(f"     Example overlapping IDs: {overlap_info['overlap_segment_ids'][:3]}")
    else:
        print("   No overlap detected")
    
    print(f"\n🔍 Exact Duplicate Segment IDs:")
    if analysis["exact_duplicate_segment_ids"]:
        print(f"   Found {len(analysis['exact_duplicate_segment_ids'])} segment_ids appearing multiple times:")
        for seg_id, count in list(analysis["exact_duplicate_segment_ids"].items())[:5]:
            print(f"     {seg_id}: appears {count} times")
        if len(analysis["exact_duplicate_segment_ids"]) > 5:
            print(f"     ... and {len(analysis['exact_duplicate_segment_ids']) - 5} more")
    else:
        print("   No exact duplicate segment_ids found")
    
    print(f"\n📋 Exact Duplicate Rows (same data, different created_at):")
    dup_rows = analysis["exact_duplicate_rows"]
    if dup_rows["count"] > 0:
        print(f"   Found {dup_rows['count']} sets of exact duplicate rows")
        print(f"   Examples:")
        for key, segs in dup_rows["examples"][:3]:
            print(f"     Segment ID: {key[0]}")
            print(f"       Appears in {len(segs)} batches:")
            for seg in segs:
                print(f"         - {seg['created_at']} (video_path: {seg['video_path']})")
    else:
        print("   No exact duplicate rows found (all rows are unique)")
    
    # Recommendation
    print(f"\n💡 Recommendation:")
    if dup_rows["count"] > 0:
        print("   ⚠️  EXACT DUPLICATES DETECTED")
        print("   → These should be merged (keep one copy per unique segment)")
        print("   → The script should deduplicate by segment_id when loading")
    elif analysis["unique_segment_ids"] < analysis["total_segments"]:
        print("   ⚠️  Some segment_ids appear multiple times")
        print("   → Check if these are truly duplicates or have different metadata")
    elif len(analysis["segment_id_overlap"]) > 0 and any(
        info["overlap_count"] > 0 for info in analysis["segment_id_overlap"].values()
    ):
        print("   ℹ️  Overlapping segment_ids between batches")
        print("   → These are likely the same segments processed in different batches")
        print("   → Should deduplicate by segment_id when loading")
    else:
        print("   ✅ No duplicates detected")
        print("   → Different batches contain different augmentations/segments")
        print("   → Should include all batches (merge by video_id)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze duplicate video_ids across created_at batches"
    )
    parser.add_argument(
        "--created-at-filter",
        type=str,
        default=DEFAULT_CREATED_AT_FILTER,
        help=f"Filter segments by created_at >= this date (default: {DEFAULT_CREATED_AT_FILTER})",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Analyze a specific video_id (if not provided, analyzes all duplicates)",
    )
    args = parser.parse_args()
    
    print(f"🔍 Analyzing duplicate video_ids")
    print(f"   Filter: created_at >= {args.created_at_filter}")
    
    conn = connect_neon()
    
    try:
        if args.video_id:
            # Analyze specific video_id
            print(f"\n📊 Analyzing video_id: {args.video_id}")
            analysis = analyze_video_id_across_batches(conn, args.video_id, args.created_at_filter)
            print_analysis(analysis)
        else:
            # Find all duplicates
            print(f"\n🔍 Finding video_ids that appear in multiple batches...")
            duplicates = find_duplicate_video_ids(conn, args.created_at_filter)
            
            if not duplicates:
                print("✅ No duplicate video_ids found!")
                return
            
            print(f"\n📊 Found {len(duplicates)} video_ids appearing in multiple batches:")
            for video_id, num_batches in duplicates:
                print(f"   {video_id}: {num_batches} batches")
            
            print(f"\n{'='*80}")
            print(f"📋 Detailed Analysis for Each Duplicate:")
            print(f"{'='*80}")
            
            for video_id, num_batches in duplicates:
                analysis = analyze_video_id_across_batches(conn, video_id, args.created_at_filter)
                print_analysis(analysis)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

