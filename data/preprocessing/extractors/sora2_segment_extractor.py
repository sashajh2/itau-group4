# Reads videos from Sora2 dataset, extracts segments, and creates metadata entries in your database.
from datetime import datetime, timezone
import os
import sys
import sqlite3
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from utils.embedding_utils import get_video_duration
from utils.config_loader import load_config
from data.preprocessing.storage.neon_writer import NeonSegmentWriter


def get_video_files_from_sora2_folder(base_dir: str) -> list:
    """
    Get all video files from the Sora2 directory.
    
    Args:
        base_dir: Base directory containing the Sora2 videos
        
    Returns:
        List of video file paths
    """
    if not os.path.exists(base_dir):
        print(f"❌ Sora2 directory not found at: {base_dir}")
        return []
    
    video_files = []
    video_extensions = ['.mp4', '.MP4', '.avi', '.mov', '.mkv']
    
    for file in os.listdir(base_dir):
        if any(file.endswith(ext) for ext in video_extensions):
            video_path = os.path.join(base_dir, file)
            video_files.append(video_path)
    
    print(f"Found {len(video_files)} video files in {base_dir}")
    return sorted(video_files)  # Sort for consistent processing order


def create_segments_from_video(video_path: str, segment_duration: float = 0.25) -> list:
    """
    Create segments from a video file by chopping it into fixed-duration pieces.
    
    Args:
        video_path: Path to the video file
        segment_duration: Duration of each segment in seconds
        
    Returns:
        List of segment dictionaries
    """
    segments = []
    
    # Get video duration
    video_duration = get_video_duration(video_path)
    if video_duration is None:
        print(f"⚠️ Could not get duration for {video_path}, skipping")
        return segments
    
    # Create segments
    current_time = 0.0
    segment_count = 0
    
    while current_time < video_duration:
        start_time = current_time
        end_time = min(current_time + segment_duration, video_duration)
        
        # Skip segments that are too short (less than 0.1 seconds)
        if end_time - start_time < 0.1:
            break
        
        segment = {
            'video_path': video_path,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'segment_id': f"{segment_count:04d}"
        }
        
        segments.append(segment)
        current_time += segment_duration
        segment_count += 1
    
    print(f"Created {len(segments)} segments from {os.path.basename(video_path)}")
    return segments


def insert_sora2_segments_to_neon(segments: list, segment_writer: NeonSegmentWriter, created_at: str):
    """
    Insert Sora2 segments into Neon Postgres.
    
    Args:
        segments: List of segment dictionaries
        segment_writer: NeonSegmentWriter instance
        created_at: Timestamp for the segments
    """
    inserted_count = 0
    failed_count = 0
    
    for segment in segments:
        video_path = segment['video_path']
        video_filename = os.path.basename(video_path)
        # Use filename without extension as video_id
        video_id = os.path.splitext(video_filename)[0]  # Remove extension
        
        # Create segment ID
        segment_id = f"{video_id}_{segment['segment_id']}"
        
        try:
            segment_writer.add(
                segment_id=segment_id,
                source="Sora2",
                video_id=video_id,
                video_path=video_path,
                start_time=float(segment['start_time']),
                duration=float(segment['duration']),
                video_label=1.0,  # video_label = 1.0 (fake, stored as float)
                audio_label=1.0,  # audio_label = 1.0 (fake, stored as float)
                audio_model="sora2",
                video_model="sora2",
                created_at=created_at,
            )
            inserted_count += 1
        except Exception as e:
            failed_count += 1
            print(f"⚠️ Error inserting segment {segment_id}: {e}")
            continue
    
    print(f"📝 Inserted {inserted_count} segments into Neon (failed: {failed_count})")


def insert_sora2_segments_to_sqlite(segments: list, db_path: str, created_at: str, source_folder: str):
    """
    Insert Sora2 segments into SQLite database (fallback).
    
    Args:
        segments: List of segment dictionaries
        db_path: Path to the SQLite database
        created_at: Timestamp for the segments
        source_folder: Path to the Sora2 folder
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    inserted_count = 0
    
    for segment in segments:
        video_path = segment['video_path']
        video_filename = os.path.basename(video_path)
        video_id = os.path.splitext(video_filename)[0]  # Remove extension
        
        # Create segment ID
        segment_id = f"{video_id}_{segment['segment_id']}"
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO segments (
                    segment_id, source, video_id, video_path, start_time, duration, 
                    video_label, audio_label, created_at, audio_model, video_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                segment_id,
                "Sora2",
                video_id,
                video_path,
                float(segment['start_time']),
                float(segment['duration']),
                1,  # video_label = 1 (fake)
                1,  # audio_label = 1 (fake)
                created_at,
                "sora2",  # audio_model = "sora2"
                "sora2"   # video_model = "sora2"
            ))
            
            inserted_count += 1
            
        except Exception as e:
            print(f"⚠️ Error inserting segment {segment_id}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    print(f"✅ Inserted {inserted_count} segments into database (SQLite fallback)")


def extract_and_insert_sora2_segments(video_root: str, created_at: str, segment_duration: float = 0.25, segment_writer: Optional[NeonSegmentWriter] = None) -> int:
    """
    Extract segments from Sora2 dataset and insert into DB.
    
    Args:
        video_root: Root directory containing the Sora2 videos
        created_at: Timestamp for the segments
        segment_duration: Duration of each segment in seconds
        segment_writer: Optional NeonSegmentWriter for Neon writes
        
    Returns:
        Number of segments inserted
    """
    print(f"Processing Sora2 dataset from: {video_root}")
    print(f"Segment duration: {segment_duration} seconds")
    print(f"Using created_at timestamp: {created_at}")
    
    # Get video files
    video_files = get_video_files_from_sora2_folder(video_root)
    if not video_files:
        print("❌ No video files found")
        return 0
    
    # Process each video
    all_segments = []
    for video_path in video_files:
        print(f"\nProcessing: {os.path.basename(video_path)}")
        segments = create_segments_from_video(video_path, segment_duration)
        all_segments.extend(segments)
    
    print(f"🧩 Created {len(all_segments)} candidate segments from videos")
    
    # Insert into database
    if all_segments:
        if segment_writer is not None:
            insert_sora2_segments_to_neon(all_segments, segment_writer, created_at)
            return len(all_segments)
        else:
            # Fallback to SQLite
            config = load_config()
            db_path = config["database"]["embedding_db_path"]
            insert_sora2_segments_to_sqlite(all_segments, db_path, created_at, video_root)
            return len(all_segments)
    
    return 0


def main():
    """Main CLI function for Sora2 segment extraction."""
    parser = argparse.ArgumentParser(description="Extract Sora2 segments and insert into database")
    parser.add_argument("--video-root", type=str, required=True, help="Root folder containing Sora2 video files")
    parser.add_argument("--created-at", type=str, default=None, help="ISO8601 timestamp to tag inserted segments")
    parser.add_argument("--segment-duration", type=float, default=0.25, help="Duration of each segment in seconds")
    
    args = parser.parse_args()
    
    created_at = args.created_at or datetime.now(timezone.utc).isoformat()
    
    try:
        num_segments = extract_and_insert_sora2_segments(
            args.video_root, 
            created_at, 
            args.segment_duration
        )
        print(f"\n🎉 Successfully processed {num_segments} Sora2 segments")
        print(f"CREATED_AT={created_at}")
        return 0
    except Exception as e:
        print(f"❌ Error processing Sora2 dataset: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

