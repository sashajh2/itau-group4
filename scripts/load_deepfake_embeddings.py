#!/usr/bin/env python3
"""
Load deepfake embedding data from Neon PostgreSQL and save as HDF5.

This script:
1. Queries segments and embeddings from Neon Postgres (created_at >= filter)
2. Groups data hierarchically by video_id and augmentations
3. Identifies source videos and classifies augmentation types
4. Saves structured data as HDF5 for efficient storage and partial loading

Usage:
    python scripts/load_deepfake_embeddings.py [--created-at-filter '2025-11-01 00:00:00'] [--output deepfake_embeddings.h5] [--dry-run]
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import psycopg2
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config

# Configuration
EMBEDDING_TABLES = {
    "openl3": "embeddings_audio_openl3",
    "hubert": "embeddings_audio_hubert",
    "senet": "embeddings_video_senet",
}

# Default filter date
DEFAULT_CREATED_AT_FILTER = "2025-11-01 00:00:00"


def connect_neon():
    """Connect to Neon Postgres database."""
    cfg = load_config()
    dsn = cfg["database"]["postgres"]["neon_database_url"]
    return psycopg2.connect(dsn)


def identify_source_video(video_paths: List[str]) -> int:
    """
    Identify the source video from list of video_paths.
    
    Priority rules:
    1. Filename ends with 'real.mp4' (no augmentation suffix)
    2. Filename is just '{video_id}.mp4' (no _p1, _p2, etc.)
    3. Fallback: alphabetically first
    
    Args:
        video_paths: List of video paths for a given video_id
        
    Returns:
        Index of source video in video_paths list
    """
    for idx, path in enumerate(video_paths):
        filename = os.path.basename(path)
        
        # Rule 1: exact match 'real.mp4'
        if filename == "real.mp4":
            return idx
        
        # Rule 2: Extract video_id and check if filename is just video_id.mp4
        # Assuming path structure: .../video_id/file.mp4
        path_parts = path.split("/")
        if len(path_parts) >= 2:
            video_id = path_parts[-2]  # Parent directory
            if filename == f"{video_id}.mp4":
                return idx
    
    # Fallback: alphabetically first
    print(f"  ⚠️  Warning: No clear source video found, using first alphabetically")
    return 0


def classify_augmentation_type(
    video_path: str, source_idx: int, video_idx: int, audio_labels: np.ndarray
) -> str:
    """
    Classify augmentation type: 'source', 'real', or 'fake'.
    
    Args:
        video_path: Path to the video
        source_idx: Index of source video
        video_idx: Current video index
        audio_labels: Array of audio labels for this augmentation
        
    Returns:
        'source', 'real', or 'fake'
    """
    if video_idx == source_idx:
        return "source"
    
    # Check if audio labels indicate fake content
    # Consider > 0.01 as threshold for fake (to handle floating point precision)
    if np.any(audio_labels > 0.01):
        return "fake"
    else:
        return "real"


def get_all_video_ids(conn: psycopg2.extensions.connection, created_at_filter: str) -> List[str]:
    """Get all unique video_ids with created_at filter."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT video_id
            FROM segments
            WHERE created_at >= %s
            ORDER BY video_id
        """, (created_at_filter,))
        return [row[0] for row in cur.fetchall()]


def fetch_video_data(
    conn: psycopg2.extensions.connection,
    video_id: str,
    created_at_filter: str,
) -> List[Dict]:
    """
    Fetch all segments and embeddings for a given video_id.
    
    Returns:
        List of dictionaries with segment data and embeddings
    """
    query = """
        SELECT 
            s.segment_id,
            s.video_path,
            s.start_time,
            s.audio_label,
            s.video_label,
            s.duration,
            e1.embedding::float4[] AS openl3_embedding,
            e2.embedding::float4[] AS hubert_embedding,
            e3.embedding::float4[] AS senet_embedding
        FROM segments s
        LEFT JOIN embeddings_audio_openl3 e1 ON s.segment_id = e1.segment_id
        LEFT JOIN embeddings_audio_hubert e2 ON s.segment_id = e2.segment_id
        LEFT JOIN embeddings_video_senet e3 ON s.segment_id = e3.segment_id
        WHERE s.video_id = %s
          AND s.created_at >= %s
        ORDER BY s.video_path, s.start_time
    """
    
    with conn.cursor() as cur:
        cur.execute(query, (video_id, created_at_filter))
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        
        results = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            # Convert embedding lists to numpy arrays
            for emb_type in ["openl3_embedding", "hubert_embedding", "senet_embedding"]:
                if row_dict[emb_type] is not None:
                    row_dict[emb_type] = np.array(row_dict[emb_type], dtype=np.float32)
            results.append(row_dict)
        
        return results


def process_video_data(
    video_id: str, raw_data: List[Dict]
) -> Optional[Dict]:
    """
    Process raw segment data into hierarchical structure for one video.
    
    Args:
        video_id: The video_id
        raw_data: List of segment dictionaries from fetch_video_data
        
    Returns:
        Dictionary with processed video data, or None if invalid
    """
    if not raw_data:
        return None
    
    # Group segments by video_path (each path is one augmentation)
    by_path: Dict[str, List[Dict]] = defaultdict(list)
    for segment in raw_data:
        by_path[segment["video_path"]].append(segment)
    
    video_paths = sorted(by_path.keys())
    num_augmentations = len(video_paths)
    
    # Check that all augmentations have the same number of segments
    segment_counts = [len(by_path[path]) for path in video_paths]
    if len(set(segment_counts)) > 1:
        print(f"  ⚠️  Warning: {video_id} has mismatched segment counts: {segment_counts}")
        # Use minimum count to ensure alignment
        num_segments = min(segment_counts)
    else:
        num_segments = segment_counts[0] if segment_counts else 0
    
    if num_segments == 0:
        print(f"  ⚠️  Warning: {video_id} has no segments, skipping")
        return None
    
    # Identify source video
    source_idx = identify_source_video(video_paths)
    
    # Detect embedding dimensions from first segment with embeddings
    emb_dims = {"openl3": None, "hubert": None, "senet": None}
    for segment in raw_data:
        if emb_dims["openl3"] is None and segment.get("openl3_embedding") is not None:
            emb_dims["openl3"] = len(segment["openl3_embedding"])
        if emb_dims["hubert"] is None and segment.get("hubert_embedding") is not None:
            emb_dims["hubert"] = len(segment["hubert_embedding"])
        if emb_dims["senet"] is None and segment.get("senet_embedding") is not None:
            emb_dims["senet"] = len(segment["senet_embedding"])
        if all(d is not None for d in emb_dims.values()):
            break
    
    # Use default dimensions if not detected (fallback)
    if emb_dims["openl3"] is None:
        emb_dims["openl3"] = 512
    if emb_dims["hubert"] is None:
        emb_dims["hubert"] = 768
    if emb_dims["senet"] is None:
        emb_dims["senet"] = 2048
    
    # Initialize arrays
    embeddings = {
        "openl3": np.zeros((num_augmentations, num_segments, emb_dims["openl3"]), dtype=np.float32),
        "hubert": np.zeros((num_augmentations, num_segments, emb_dims["hubert"]), dtype=np.float32),
        "senet": np.zeros((num_augmentations, num_segments, emb_dims["senet"]), dtype=np.float32),
    }
    labels = {
        "audio": np.zeros((num_augmentations, num_segments), dtype=np.float32),
        "video": np.zeros((num_augmentations, num_segments), dtype=np.float32),
    }
    segment_ids = []
    augmentation_types = []
    
    # Fill arrays
    for aug_idx, video_path in enumerate(video_paths):
        # Sort by start_time to ensure temporal alignment
        segments = sorted(by_path[video_path], key=lambda x: x.get("start_time", 0.0))
        
        # Truncate if needed (use first num_segments)
        segments = segments[:num_segments]
        
        for seg_idx, segment in enumerate(segments):
            # Embeddings (handle missing embeddings)
            if segment.get("openl3_embedding") is not None:
                emb = segment["openl3_embedding"]
                if len(emb) == emb_dims["openl3"]:
                    embeddings["openl3"][aug_idx, seg_idx, :] = emb
            if segment.get("hubert_embedding") is not None:
                emb = segment["hubert_embedding"]
                if len(emb) == emb_dims["hubert"]:
                    embeddings["hubert"][aug_idx, seg_idx, :] = emb
            if segment.get("senet_embedding") is not None:
                emb = segment["senet_embedding"]
                if len(emb) == emb_dims["senet"]:
                    embeddings["senet"][aug_idx, seg_idx, :] = emb
            
            # Labels
            labels["audio"][aug_idx, seg_idx] = float(segment.get("audio_label", 0.0))
            labels["video"][aug_idx, seg_idx] = float(segment.get("video_label", 0.0))
            
            # Segment IDs (only store for first augmentation to save space)
            if aug_idx == 0:
                segment_ids.append(segment["segment_id"])
        
        # Classify augmentation type
        aug_type = classify_augmentation_type(
            video_path, source_idx, aug_idx, labels["audio"][aug_idx, :]
        )
        augmentation_types.append(aug_type)
    
    # Count real/fake augmentations
    num_real = sum(1 for t in augmentation_types if t == "real")
    num_fake = sum(1 for t in augmentation_types if t == "fake")
    
    # Calculate total duration (from first augmentation)
    total_duration = sum(seg.get("duration", 0.0) for seg in by_path[video_paths[0]][:num_segments])
    
    return {
        "num_segments": num_segments,
        "num_augmentations": num_augmentations,
        "num_real_augmentations": num_real,
        "num_fake_augmentations": num_fake,
        "duration": total_duration,
        "augmentation_info": {
            "video_paths": video_paths,
            "types": augmentation_types,
            "source_idx": source_idx,
        },
        "embeddings": embeddings,
        "labels": labels,
        "segment_ids": segment_ids,
    }


def save_to_hdf5(data: Dict, filename: str = "deepfake_embeddings.h5"):
    """
    Save the data structure to HDF5 format.
    
    HDF5 structure:
    /metadata/
        created_at_filter (attr)
        embedding_types (dataset)
        video_ids (dataset)
        total_videos (attr)
        total_segments (attr)
        date_created (attr)
    
    /videos/{video_id}/
        metadata/
            num_segments (attr)
            num_augmentations (attr)
            num_real_augmentations (attr)
            num_fake_augmentations (attr)
            duration (attr)
        augmentation_info/
            video_paths (dataset)
            types (dataset)
            source_idx (attr)
        embeddings/
            openl3 (dataset: shape [num_augs, num_segs, emb_dim])
            hubert (dataset)
            senet (dataset)
        labels/
            audio (dataset: shape [num_augs, num_segs])
            video (dataset)
        segment_ids (dataset)
    
    /models/  (empty groups for future use)
    /results/ (empty groups for future use)
    """
    print(f"\n💾 Saving to {filename}...")
    
    with h5py.File(filename, "w") as f:
        # Save metadata
        meta_grp = f.create_group("metadata")
        meta_grp.attrs["created_at_filter"] = data["metadata"]["created_at_filter"]
        meta_grp.attrs["total_videos"] = data["metadata"]["total_videos"]
        meta_grp.attrs["total_segments"] = data["metadata"]["total_segments"]
        meta_grp.attrs["date_created"] = str(data["metadata"]["date_created"])
        meta_grp.create_dataset(
            "embedding_types",
            data=np.array([s.encode() for s in data["metadata"]["embedding_types"]]),
        )
        meta_grp.create_dataset(
            "video_ids",
            data=np.array([s.encode() for s in data["metadata"]["video_ids"]]),
        )
        
        # Save videos
        videos_grp = f.create_group("videos")
        for video_id, video_data in tqdm(
            data["videos"].items(), desc="Saving videos", leave=False
        ):
            # Create safe group name (replace / with _)
            safe_video_id = video_id.replace("/", "_")
            vid_grp = videos_grp.create_group(safe_video_id)
            vid_grp.attrs["original_video_id"] = video_id  # Store original
            
            # Metadata
            vid_grp.attrs["num_segments"] = video_data["num_segments"]
            vid_grp.attrs["num_augmentations"] = video_data["num_augmentations"]
            vid_grp.attrs["num_real_augmentations"] = video_data["num_real_augmentations"]
            vid_grp.attrs["num_fake_augmentations"] = video_data["num_fake_augmentations"]
            vid_grp.attrs["duration"] = video_data["duration"]
            
            # Augmentation info
            aug_grp = vid_grp.create_group("augmentation_info")
            aug_grp.create_dataset(
                "video_paths",
                data=np.array([s.encode() for s in video_data["augmentation_info"]["video_paths"]]),
            )
            aug_grp.create_dataset(
                "types",
                data=np.array([s.encode() for s in video_data["augmentation_info"]["types"]]),
            )
            aug_grp.attrs["source_idx"] = video_data["augmentation_info"]["source_idx"]
            
            # Embeddings (with compression!)
            emb_grp = vid_grp.create_group("embeddings")
            for emb_type in ["openl3", "hubert", "senet"]:
                emb_grp.create_dataset(
                    emb_type,
                    data=video_data["embeddings"][emb_type],
                    compression="gzip",
                    compression_opts=4,  # Balance between speed and compression
                )
            
            # Labels
            lbl_grp = vid_grp.create_group("labels")
            lbl_grp.create_dataset("audio", data=video_data["labels"]["audio"])
            lbl_grp.create_dataset("video", data=video_data["labels"]["video"])
            
            # Segment IDs
            vid_grp.create_dataset(
                "segment_ids",
                data=np.array([s.encode() for s in video_data["segment_ids"]]),
            )
        
        # Create empty groups for future use
        f.create_group("models")
        f.create_group("results")
    
    file_size_gb = os.path.getsize(filename) / 1e9
    print(f"✅ Data saved to {filename}")
    print(f"📊 File size: {file_size_gb:.2f} GB")


def load_from_hdf5(
    filename: str = "deepfake_embeddings.h5", video_ids: Optional[List[str]] = None
) -> Dict:
    """
    Load data from HDF5. Can optionally load only specific video_ids.
    
    Args:
        filename: Path to HDF5 file
        video_ids: Optional list of video_ids to load (None = load all)
    
    Returns:
        data dictionary with same structure as original
    """
    data = {
        "metadata": {},
        "videos": {},
        "models": {"pca": {}, "umap": {}, "tsne": None},
        "results": {},
    }
    
    with h5py.File(filename, "r") as f:
        # Load metadata
        meta_grp = f["metadata"]
        data["metadata"] = {
            "created_at_filter": meta_grp.attrs["created_at_filter"].decode()
            if isinstance(meta_grp.attrs["created_at_filter"], bytes)
            else meta_grp.attrs["created_at_filter"],
            "total_videos": meta_grp.attrs["total_videos"],
            "total_segments": meta_grp.attrs["total_segments"],
            "date_created": meta_grp.attrs["date_created"].decode()
            if isinstance(meta_grp.attrs["date_created"], bytes)
            else meta_grp.attrs["date_created"],
            "embedding_types": [s.decode() for s in meta_grp["embedding_types"][:]],
            "video_ids": [s.decode() for s in meta_grp["video_ids"][:]],
        }
        
        # Filter video_ids if specified
        if video_ids is not None:
            video_ids_to_load = video_ids
        else:
            video_ids_to_load = data["metadata"]["video_ids"]
        
        # Load videos
        videos_grp = f["videos"]
        for video_id in video_ids_to_load:
            safe_video_id = video_id.replace("/", "_")
            if safe_video_id not in videos_grp:
                print(f"⚠️  Warning: {video_id} not found in file")
                continue
            
            vid_grp = videos_grp[safe_video_id]
            
            data["videos"][video_id] = {
                "num_segments": vid_grp.attrs["num_segments"],
                "num_augmentations": vid_grp.attrs["num_augmentations"],
                "num_real_augmentations": vid_grp.attrs["num_real_augmentations"],
                "num_fake_augmentations": vid_grp.attrs["num_fake_augmentations"],
                "duration": vid_grp.attrs["duration"],
                "augmentation_info": {
                    "video_paths": [
                        s.decode() for s in vid_grp["augmentation_info/video_paths"][:]
                    ],
                    "types": [
                        s.decode() for s in vid_grp["augmentation_info/types"][:]
                    ],
                    "source_idx": vid_grp["augmentation_info"].attrs["source_idx"],
                },
                "embeddings": {
                    "openl3": vid_grp["embeddings/openl3"][:],
                    "hubert": vid_grp["embeddings/hubert"][:],
                    "senet": vid_grp["embeddings/senet"][:],
                },
                "labels": {
                    "audio": vid_grp["labels/audio"][:],
                    "video": vid_grp["labels/video"][:],
                },
                "segment_ids": [
                    s.decode() for s in vid_grp["segment_ids"][:]
                ],
            }
    
    return data


def validate_data(data: Dict):
    """Validate the loaded data structure."""
    assert "metadata" in data
    assert "videos" in data
    assert len(data["metadata"]["video_ids"]) == len(data["videos"])
    
    for video_id, video_data in data["videos"].items():
        # Check shape consistency
        num_augs, num_segs, _ = video_data["embeddings"]["openl3"].shape
        assert video_data["num_augmentations"] == num_augs
        assert video_data["num_segments"] == num_segs
        
        # Check all embeddings have same shape
        assert video_data["embeddings"]["hubert"].shape[0] == num_augs
        assert video_data["embeddings"]["senet"].shape[0] == num_augs
        
        # Check labels shape
        assert video_data["labels"]["audio"].shape == (num_augs, num_segs)
        
        print(f"  ✓ {video_id}: {num_augs} augs, {num_segs} segments")


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
        default="deepfake_embeddings.h5",
        help="Output HDF5 filename (default: deepfake_embeddings.h5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test queries without loading all data (just count videos)",
    )
    args = parser.parse_args()
    
    print(f"📊 Loading deepfake embeddings from Neon")
    print(f"   Filter: created_at >= {args.created_at_filter}")
    print(f"   Output: {args.output}")
    
    # Connect to database
    print(f"\n🔗 Connecting to Neon...")
    conn = connect_neon()
    
    try:
        # Get all video_ids
        print(f"\n📥 Fetching video_ids...")
        video_ids = get_all_video_ids(conn, args.created_at_filter)
        print(f"   Found {len(video_ids)} unique video_ids")
        
        if args.dry_run:
            print(f"\n✅ Dry run complete. Would process {len(video_ids)} videos.")
            return
        
        # Process each video
        print(f"\n🔄 Processing {len(video_ids)} videos...")
        videos_data = {}
        total_segments = 0
        
        for video_id in tqdm(video_ids, desc="Processing videos"):
            raw_data = fetch_video_data(conn, video_id, args.created_at_filter)
            processed = process_video_data(video_id, raw_data)
            
            if processed is not None:
                videos_data[video_id] = processed
                total_segments += processed["num_segments"]
        
        print(f"\n✅ Loaded {len(videos_data)} videos with {total_segments} total segments")
        
        # Build final data structure
        data = {
            "metadata": {
                "created_at_filter": args.created_at_filter,
                "embedding_types": list(EMBEDDING_TABLES.keys()),
                "video_ids": sorted(videos_data.keys()),
                "total_videos": len(videos_data),
                "total_segments": total_segments,
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

