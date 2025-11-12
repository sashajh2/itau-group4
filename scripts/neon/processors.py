"""
Video processing logic for deepfake embeddings.

This module provides functions to process raw segment data into structured video data,
including source identification and augmentation classification.
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from .validation import check_video_completeness


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


def process_video_data(
    video_id: str, 
    raw_data: List[Dict], 
    require_complete: bool = False, 
    dataset: str = "avdeepfake1m",
    exclude_incomplete_augmentations: bool = False
) -> Optional[Dict]:
    """
    Process raw segment data into hierarchical structure for one video.
    
    Args:
        video_id: The video_id
        raw_data: List of segment dictionaries from fetch_video_data
        require_complete: If True, skip videos without all embeddings
        dataset: Dataset name ('avdeepfake1m' or 'shareveo3')
        exclude_incomplete_augmentations: If True, exclude entire augmentations if any segment is missing
        
    Returns:
        Dictionary with processed video data, or None if invalid/incomplete
    """
    if not raw_data:
        return None
    
    # Check completeness if required
    if require_complete:
        is_complete, stats = check_video_completeness(raw_data)
        if not is_complete:
            return None
    
    # Detect dataset from source field if not provided
    if dataset is None:
        sources = set(seg.get("source") for seg in raw_data if seg.get("source"))
        if "ShareVeo3" in sources:
            dataset = "shareveo3"
        else:
            dataset = "avdeepfake1m"
    
    # Group segments by video_path (each path is one augmentation)
    by_path: Dict[str, List[Dict]] = defaultdict(list)
    for segment in raw_data:
        by_path[segment["video_path"]].append(segment)
    
    # If exclude_incomplete_augmentations is True, remove augmentations with missing embeddings
    if exclude_incomplete_augmentations:
        complete_paths = []
        for path, segments in by_path.items():
            # Check if all segments in this augmentation have all 3 embeddings
            all_complete = all(
                seg.get("openl3_embedding") is not None
                and seg.get("hubert_embedding") is not None
                and seg.get("senet_embedding") is not None
                for seg in segments
            )
            if all_complete:
                complete_paths.append(path)
            else:
                incomplete_count = sum(
                    1 for seg in segments
                    if not (seg.get("openl3_embedding") is not None
                           and seg.get("hubert_embedding") is not None
                           and seg.get("senet_embedding") is not None)
                )
                print(f"  ⚠️  {video_id}: Excluding augmentation {path} ({incomplete_count} segments missing embeddings)")
        
        # Only keep complete augmentations
        by_path = {path: by_path[path] for path in complete_paths}
        if not by_path:
            print(f"  ⚠️  {video_id}: All augmentations excluded, skipping video")
            return None
    
    video_paths = sorted(by_path.keys())
    num_augmentations = len(video_paths)
    
    # ShareVeo3 should have exactly 1 augmentation
    if dataset == "shareveo3" and num_augmentations != 1:
        print(f"  ⚠️  Warning: ShareVeo3 video {video_id} has {num_augmentations} augmentations (expected 1)")
    
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
    
    # Identify source video (only for AVDeepfake1M)
    if dataset == "avdeepfake1m":
        source_idx = identify_source_video(video_paths)
    else:
        # ShareVeo3: source is the single video itself
        source_idx = 0
    
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
        if dataset == "shareveo3":
            # ShareVeo3 is always fully synthetic (fake)
            aug_type = "fake"
        else:
            aug_type = classify_augmentation_type(
                video_path, source_idx, aug_idx, labels["audio"][aug_idx, :]
            )
        augmentation_types.append(aug_type)
    
    # Count real/fake augmentations
    # For ShareVeo3, source is counted as fake (it's synthetic)
    if dataset == "shareveo3":
        num_real = 0
        num_fake = 1
    else:
        num_real = sum(1 for t in augmentation_types if t in ["source", "real"])
        num_fake = sum(1 for t in augmentation_types if t == "fake")
    
    # Calculate total duration (from first augmentation)
    total_duration = sum(seg.get("duration", 0.0) for seg in by_path[video_paths[0]][:num_segments])
    
    # Get created_at (for AVDeepfake1M)
    created_at_val = None
    if dataset == "avdeepfake1m" and raw_data:
        created_at_val = raw_data[0].get("created_at")
        if created_at_val is not None:
            created_at_val = str(created_at_val)
    
    return {
        "dataset": dataset,
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
        "created_at": created_at_val,
    }
