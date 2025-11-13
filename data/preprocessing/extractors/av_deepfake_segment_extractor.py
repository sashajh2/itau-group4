"""
Segment extraction for AVDeepfake1M dataset.

This module provides:
- Metadata loading from JSON files
- Full segment extraction (metadata-based, no partials)
- Uniform segment extraction (fixed-length with soft labels)
- Database insertion (Neon Postgres or SQLite fallback)
- Debug utilities for testing uniform segmentation
"""
from datetime import datetime, timezone
import os
import json
import random
import sqlite3
import argparse
import pandas as pd
from typing import Optional
from utils.embedding_utils import get_video_duration, sample_real_segment
from utils.config_loader import load_config
from ..storage.neon_writer import NeonSegmentWriter

def load_video_metadata(base_dir: str) -> pd.DataFrame:
    entries = []

    print(f"Walking through: {base_dir}")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                mp4_path = json_path.replace('.json', '.mp4')
                if not os.path.exists(mp4_path):
                    continue  # Skip if no matching mp4

                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    audio_model = data.get('audio_model')
                    video_model = data.get('video_model')
                    audio_label = 0 if audio_model is None else 1
                    video_label = 0 if video_model is None else 1

                    audio_fake_segments = data.get('audio_fake_segments', None)
                    visual_fake_segments = data.get('visual_fake_segments', None)

                    entries.append({
                        'video_path': mp4_path,
                        'json_path': json_path,
                        'audio_label': audio_label,
                        'video_label': video_label,
                        'audio_fake_segments': audio_fake_segments if audio_fake_segments else None,
                        'visual_fake_segments': visual_fake_segments if visual_fake_segments else None,
                        'source_folder': root,
                        'audio_model': audio_model,
                        'video_model': video_model
                    })

                except Exception as e:
                    print(f"Error reading {json_path}: {e}")

    df = pd.DataFrame(entries)
    return df

def generate_segment_metadata(video_metadata_df: pd.DataFrame, real_clip_duration_bounds=(0.1, 0.46)) -> pd.DataFrame:
    segment_rows = []
    
    processed_videos = 0

    for _, row in video_metadata_df.iterrows():
        # if both fake
        if row['audio_label'] == 1 and row['video_label'] == 1:
            for seg in row['audio_fake_segments']:
                start, end = seg[0], seg[1]
                segment_rows.append({
                    'audio_label': 1.0,  # Store as float for consistency
                    'video_label': 1.0,
                    'overall_label': 1,
                    'video_path': row['video_path'],
                    'json_path': row['json_path'],
                    'source_folder': row['source_folder'],
                    'segment_start': start,
                    'segment_end': end,
                    'audio_model': row['audio_model'],
                    'video_model': row['video_model']
                })

        # audio fake, video real
        elif row['audio_label'] == 1 and row['video_label'] == 0:
            for seg in row['audio_fake_segments']:
                start, end = seg[0], seg[1]
                segment_rows.append({
                    'audio_label': 1.0,
                    'video_label': 0.0,
                    'overall_label': 1,
                    'video_path': row['video_path'],
                    'json_path': row['json_path'],
                    'source_folder': row['source_folder'],
                    'segment_start': start,
                    'segment_end': end,
                    'audio_model': row['audio_model'],
                    'video_model': row['video_model']
                })
        # audio real, video fake
        elif row['audio_label'] == 0 and row['video_label'] == 1:
            for seg in row['visual_fake_segments']:
                start, end = seg[0], seg[1]
                segment_rows.append({
                    'audio_label': 0.0,
                    'video_label': 1.0,
                    'overall_label': 1,
                    'video_path': row['video_path'],
                    'json_path': row['json_path'],
                    'source_folder': row['source_folder'],
                    'segment_start': start,
                    'segment_end': end,
                    'audio_model': row['audio_model'],
                    'video_model': row['video_model']
                })
        # both real
        else:
            # randomly sample multiple segments of the real video (5 segments)
            real_duration = get_video_duration(row['video_path'])
            if real_duration is None:
                print(f"⚠️  Skipping video due to duration error: {row['video_path']}")
                continue  # skip if there's an error loading
            
            num_real_segments = 5
            for _ in range(num_real_segments):
                segment_length = round(random.uniform(*real_clip_duration_bounds), 2)
                start, end = sample_real_segment(real_duration, segment_length)
                segment_rows.append({
                    'audio_label': 0.0,
                    'video_label': 0.0,
                    'overall_label': 0,
                    'video_path': row['video_path'],
                    'json_path': row['json_path'],
                    'source_folder': row['source_folder'],
                    'segment_start': start,
                    'segment_end': end,
                    'audio_model': row['audio_model'],
                    'video_model': row['video_model']
                })
        
        processed_videos += 1

    print(f"📊 Video processing summary:")
    print(f"   Processed videos: {processed_videos}")

    return pd.DataFrame(segment_rows)


def _interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Calculate overlap duration between two time intervals."""
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


def _proportion_fake(segment_start: float, segment_end: float, fake_segments: Optional[list]) -> float:
    """
    Calculate the proportion of fake content within a segment.
    
    Args:
        segment_start: Start time of the segment
        segment_end: End time of the segment
        fake_segments: List of [start, end] tuples for fake segments, or None
        
    Returns:
        Proportion of fake content (0.0 to 1.0)
    """
    if not fake_segments:
        return 0.0
    
    total = segment_end - segment_start
    if total <= 0:
        return 0.0
    
    overlap = 0.0
    for seg in fake_segments:
        try:
            fs, fe = float(seg[0]), float(seg[1])
            overlap += _interval_overlap(segment_start, segment_end, fs, fe)
        except (ValueError, TypeError, IndexError):
            continue
    
    return min(1.0, max(0.0, overlap / total))


def generate_uniform_segments(
    video_metadata_df: pd.DataFrame, 
    segment_length: float, 
    stride: Optional[float] = None
) -> pd.DataFrame:
    """
    Generate uniform-length segments from videos with soft labels.
    
    Args:
        video_metadata_df: DataFrame with video metadata (from load_video_metadata)
        segment_length: Length of each segment in seconds
        stride: Stride between segments (default: segment_length, i.e., no overlap)
        
    Returns:
        DataFrame with segment metadata. audio_label and video_label contain soft labels (0.0-1.0).
    """
    if stride is None:
        stride = segment_length
    
    segment_rows = []
    processed_videos = 0
    
    for _, row in video_metadata_df.iterrows():
        duration = get_video_duration(row['video_path'])
        if not duration or duration <= 0:
            print(f"⚠️  Skipping video due to duration error: {row['video_path']}")
            continue
        
        start = 0.0
        while start + 1e-6 < duration:
            end = min(duration, start + segment_length)
            
            # Drop extremely short tail segments
            if end - start <= 0.02:
                break
            
            # Calculate soft labels (proportion of fake content) - store directly as labels
            audio_soft = _proportion_fake(start, end, row.get('audio_fake_segments'))
            video_soft = _proportion_fake(start, end, row.get('visual_fake_segments'))
            
            # Use model from original video metadata (model assignment handled in post-processing)
            audio_model = row['audio_model']
            video_model = row['video_model']
            
            segment_rows.append({
                'audio_label': audio_soft,  # Store soft label directly (0.0 to 1.0)
                'video_label': video_soft,  # Store soft label directly (0.0 to 1.0)
                'video_path': row['video_path'],
                'json_path': row['json_path'],
                'source_folder': row['source_folder'],
                'segment_start': start,
                'segment_end': end,
                'audio_model': audio_model,
                'video_model': video_model
            })
            
            if end >= duration:
                break
            start += stride
        
        processed_videos += 1
    
    print(f"📊 Uniform segmentation summary: processed_videos={processed_videos}, segments={len(segment_rows)}")
    return pd.DataFrame(segment_rows)


def extract_and_insert_uniform_segments(
    video_root: str, 
    created_at: str, 
    segment_length: float, 
    stride: Optional[float] = None,
    segment_writer: Optional[NeonSegmentWriter] = None,
    limit: Optional[int] = None
) -> int:
    """
    Extract uniform-length segments from videos and insert into DB.
    
    Args:
        video_root: Root directory containing video files
        created_at: ISO8601 timestamp for created_at field
        segment_length: Length of each uniform segment in seconds
        stride: Stride between segments (default: segment_length)
        segment_writer: Optional NeonSegmentWriter for Neon writes
        limit: Optional limit on number of segments to process (for testing)
        
    Returns:
        Number of segments inserted
    """
    metadata_df = load_video_metadata(video_root)
    segment_df = generate_uniform_segments(
        metadata_df, 
        segment_length=segment_length, 
        stride=stride
    )
    print(f"🧩 Created {len(segment_df)} candidate segments (uniform segmentation)")
    
    if segment_writer is not None:
        actual_inserted = insert_segments_to_neon(segment_df, segment_writer, created_at, limit=limit)
    else:
        # Fallback to SQLite for backwards compatibility
        config = load_config()
        db_path = config["database"]["embedding_db_path"]
        if limit is not None:
            segment_df = segment_df.head(limit)
            print(f"🧪 TEST MODE: Limiting to {limit} segments")
        actual_inserted = insert_segments_to_sqlite_fallback(segment_df, db_path, created_at)
    
    return actual_inserted


def insert_segments_to_neon(segment_metadata_df: pd.DataFrame, segment_writer: NeonSegmentWriter, created_at: str, limit: Optional[int] = None):
    # Check if there are any segments to insert
    if len(segment_metadata_df) == 0:
        print("⚠️  No segments to insert - skipping database insertion")
        return 0
    
    # Apply limit for testing purposes
    if limit is not None:
        segment_metadata_df = segment_metadata_df.head(limit)
        print(f"🧪 TEST MODE: Limiting to {limit} segments")
    
    successful_inserts = 0
    failed_inserts = 0
    
    print(f"🔍 Starting insertion of {len(segment_metadata_df)} segments...")
    
    for i, (_, row) in enumerate(segment_metadata_df.iterrows()):
        try:
            video_path_parts = row['video_path'].split('/')
            parent_folder = video_path_parts[-3]  # e.g., '0N1oA9LUEc4'
            child_folder = video_path_parts[-2]   # e.g., '00008'
            video_name = os.path.basename(row['video_path']).replace('.mp4', '')
            segment_ms = int(row['segment_start'] * 1000)

            segment_id = f"{parent_folder}/{child_folder}/{video_name}_{segment_ms}_{int(row['segment_end'] * 1000)}"
            video_id = f"{parent_folder}/{child_folder}"

            # Debug: Print every 100th segment being processed
            if i % 100 == 0:
                print(f"🔍 Processing segment {i}/{len(segment_metadata_df)}: {segment_id}")

            segment_writer.add(
                segment_id=segment_id,
                source="AVDeepfake1M",
                video_id=video_id,
                video_path=row['video_path'],
                start_time=float(row['segment_start']),
                duration=float(row['segment_end'] - row['segment_start']),
                video_label=float(row['video_label']),  # Ensure float type
                audio_label=float(row['audio_label']),  # Ensure float type
                audio_model=row['audio_model'],
                video_model=row['video_model'],
                created_at=created_at,
            )
            successful_inserts += 1
        except Exception as e:
            failed_inserts += 1
            print(f"❌ Failed to insert segment {i}: {e}")
            print(f"   Row data: {dict(row)}")
            if 'video_path_parts' in locals():
                print(f"   Video path parts: {video_path_parts}")
            if 'segment_id' in locals():
                print(f"   Generated segment_id: {segment_id}")
    
    print(f"📊 Insert results: {successful_inserts} successful, {failed_inserts} failed")
    
    return successful_inserts


def extract_and_insert_segments(
    video_root: str, 
    created_at: str, 
    segment_writer: Optional[NeonSegmentWriter] = None, 
    limit: Optional[int] = None,
    segmentation_mode: str = "full",
    uniform_segment_length: Optional[float] = None,
    uniform_stride: Optional[float] = None
) -> int:
    """
    Extract segments from a video root and insert into DB using provided created_at.
    
    If segment_writer is provided, writes to Neon. Otherwise falls back to SQLite for backwards compatibility.
    
    Args:
        video_root: Root directory containing video files
        created_at: ISO8601 timestamp for created_at field
        segment_writer: Optional NeonSegmentWriter for Neon writes
        limit: Optional limit on number of segments to process (for testing)
        segmentation_mode: "full" (no partials) or "uniform"
        uniform_segment_length: Length of uniform segments (required if mode="uniform")
        uniform_stride: Stride between uniform segments (default: segment_length)

    Returns number of segments inserted.
    """
    if segmentation_mode == "uniform":
        if uniform_segment_length is None:
            raise ValueError("uniform_segment_length is required when segmentation_mode='uniform'")
        return extract_and_insert_uniform_segments(
            video_root,
            created_at,
            segment_length=uniform_segment_length,
            stride=uniform_stride,
            segment_writer=segment_writer,
            limit=limit
        )
    elif segmentation_mode == "full":
        # Default: metadata-based segmentation with fully real/fake segments
        metadata_df = load_video_metadata(video_root)
        segment_df = generate_segment_metadata(metadata_df)
        print(f"🧩 Created {len(segment_df)} candidate segments from metadata")
        
        if segment_writer is not None:
            actual_inserted = insert_segments_to_neon(segment_df, segment_writer, created_at, limit=limit)
        else:
            # Fallback to SQLite for backwards compatibility
            config = load_config()
            db_path = config["database"]["embedding_db_path"]
            if limit is not None:
                segment_df = segment_df.head(limit)
                print(f"🧪 TEST MODE: Limiting to {limit} segments")
            actual_inserted = insert_segments_to_sqlite_fallback(segment_df, db_path, created_at)
        
        return actual_inserted
    else:
        raise ValueError("segmentation_mode must be one of {'full','uniform'}")


def insert_segments_to_sqlite_fallback(segment_metadata_df: pd.DataFrame, db_path: str, created_at: str):
    """Fallback SQLite insertion for backwards compatibility."""
    # Check if there are any segments to insert
    if len(segment_metadata_df) == 0:
        print("⚠️  No segments to insert - skipping database insertion")
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    successful_inserts = 0
    failed_inserts = 0
    
    print(f"🔍 Starting insertion of {len(segment_metadata_df)} segments (SQLite fallback)...")
    
    for i, (_, row) in enumerate(segment_metadata_df.iterrows()):
        try:
            video_path_parts = row['video_path'].split('/')
            parent_folder = video_path_parts[-3]  # e.g., '0N1oA9LUEc4'
            child_folder = video_path_parts[-2]   # e.g., '00008'
            video_name = os.path.basename(row['video_path']).replace('.mp4', '')
            segment_ms = int(row['segment_start'] * 1000)

            segment_id = f"{parent_folder}/{child_folder}/{video_name}_{segment_ms}_{int(row['segment_end'] * 1000)}"
            video_id = f"{parent_folder}/{child_folder}"

            cursor.execute("""
                INSERT OR REPLACE INTO segments (
                    segment_id, source, video_id, video_path, start_time, duration, video_label, audio_label, created_at, audio_model, video_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                segment_id,
                "AVDeepfake1M",
                video_id,
                row['video_path'],
                float(row['segment_start']),
                float(row['segment_end'] - row['segment_start']),
                row['video_label'],
                row['audio_label'],
                created_at,
                row['audio_model'],
                row['video_model']
            ))
            successful_inserts += 1
        except Exception as e:
            failed_inserts += 1
            print(f"❌ Failed to insert segment {i}: {e}")
    
    print(f"📊 Insert results: {successful_inserts} successful, {failed_inserts} failed")

    conn.commit()
    conn.close()
    
    return successful_inserts

def main():
    """
    Legacy CLI entry point for SQLite insertion (metadata-based segmentation only).
    
    Note: For production use, prefer av_deepfake_batch_pipeline.py which handles
    Neon insertion and supports both 'full' and 'uniform' segmentation modes.
    """
    parser = argparse.ArgumentParser(
        description="Extract segments from videos and insert into SQLite (legacy mode). "
                    "For Neon insertion and uniform segmentation, use av_deepfake_batch_pipeline.py instead."
    )
    parser.add_argument("--video-root", type=str, default="./data/temp_video_extracted/AV1M/extracted/train/lrs3", 
                       help="Root folder containing extracted videos")
    parser.add_argument("--created-at", type=str, default=None, 
                       help="ISO8601 timestamp to tag inserted segments")
    args = parser.parse_args()

    config = load_config()
    db_path = config["database"]["embedding_db_path"]

    created_at = args.created_at or datetime.now(timezone.utc).isoformat()

    print("Loading video metadata...")
    metadata_df = load_video_metadata(args.video_root)
    print(f"Video metadata loaded: {len(metadata_df)} videos")

    print("Generating segment metadata (full segments only)...")
    segment_df = generate_segment_metadata(metadata_df)
    print(f"Segment metadata generated: {len(segment_df)} segments")

    print(f"Inserting {len(segment_df)} segments into SQLite with created_at={created_at}...")
    num_inserted = insert_segments_to_sqlite_fallback(segment_df, db_path, created_at)
    print(f"✅ Done! Inserted {num_inserted} segments")
    print(f"CREATED_AT={created_at}")


if __name__ == "__main__":
    main()


