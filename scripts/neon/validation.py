"""
Validation functions for deepfake embedding data.

This module provides functions to check data completeness and validate data structures.
"""

from typing import Dict, List, Tuple


def check_video_completeness(raw_data: List[Dict]) -> Tuple[bool, Dict[str, int]]:
    """
    Check if a video has complete embeddings (all 3 types for all segments).
    
    Args:
        raw_data: List of segment dictionaries
        
    Returns:
        Tuple of (is_complete, stats_dict) where stats_dict contains counts
    """
    if not raw_data:
        return False, {"total_segments": 0, "openl3": 0, "hubert": 0, "senet": 0}
    
    stats = {
        "total_segments": len(raw_data),
        "openl3": 0,
        "hubert": 0,
        "senet": 0,
    }
    
    for segment in raw_data:
        if segment.get("openl3_embedding") is not None:
            stats["openl3"] += 1
        if segment.get("hubert_embedding") is not None:
            stats["hubert"] += 1
        if segment.get("senet_embedding") is not None:
            stats["senet"] += 1
    
    # Complete if all segments have all 3 embedding types
    is_complete = (
        stats["openl3"] == stats["total_segments"]
        and stats["hubert"] == stats["total_segments"]
        and stats["senet"] == stats["total_segments"]
    )
    
    return is_complete, stats


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

