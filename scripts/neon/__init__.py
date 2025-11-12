"""
Neon database operations for loading deepfake embeddings.

This package provides:
- Database queries for fetching segments and embeddings
- Video processing logic (source identification, augmentation classification)
- HDF5 I/O operations
- Validation functions
"""

from .queries import connect_neon, get_all_video_ids, analyze_video_id_distribution, fetch_video_data
from .processors import identify_source_video, classify_augmentation_type, process_video_data
from .hdf5_io import save_to_hdf5, load_from_hdf5
from .validation import check_video_completeness, validate_data

__all__ = [
    "connect_neon",
    "get_all_video_ids",
    "analyze_video_id_distribution",
    "fetch_video_data",
    "identify_source_video",
    "classify_augmentation_type",
    "process_video_data",
    "save_to_hdf5",
    "load_from_hdf5",
    "check_video_completeness",
    "validate_data",
]

