"""
HDF5 I/O operations for deepfake embedding data.

This module provides functions to save and load structured embedding data to/from HDF5 format.
"""

import os
from typing import Dict, List, Optional

import h5py
import numpy as np
from tqdm import tqdm


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
        if "datasets" in data["metadata"]:
            meta_grp.create_dataset(
                "datasets",
                data=np.array([s.encode() for s in data["metadata"]["datasets"]]),
            )
        if "avdeepfake1m_filter" in data["metadata"]:
            meta_grp.attrs["avdeepfake1m_filter"] = data["metadata"]["avdeepfake1m_filter"]
        if "created_at_filter" in data["metadata"]:
            meta_grp.attrs["created_at_filter"] = data["metadata"]["created_at_filter"]
        meta_grp.attrs["total_videos"] = data["metadata"]["total_videos"]
        meta_grp.attrs["total_segments"] = data["metadata"].get("total_segments", data["metadata"].get("total_embeddings", 0))
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
            vid_grp.attrs["dataset"] = video_data.get("dataset", "avdeepfake1m")
            vid_grp.attrs["num_segments"] = video_data["num_segments"]
            vid_grp.attrs["num_augmentations"] = video_data["num_augmentations"]
            vid_grp.attrs["num_real_augmentations"] = video_data["num_real_augmentations"]
            vid_grp.attrs["num_fake_augmentations"] = video_data["num_fake_augmentations"]
            vid_grp.attrs["duration"] = video_data["duration"]
            if video_data.get("created_at") is not None:
                vid_grp.attrs["created_at"] = video_data["created_at"]
            
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
        metadata_dict = {
            "total_videos": meta_grp.attrs["total_videos"],
            "total_segments": meta_grp.attrs.get("total_segments", 0),
            "date_created": meta_grp.attrs["date_created"].decode()
            if isinstance(meta_grp.attrs["date_created"], bytes)
            else meta_grp.attrs["date_created"],
            "embedding_types": [s.decode() for s in meta_grp["embedding_types"][:]],
            "video_ids": [s.decode() for s in meta_grp["video_ids"][:]],
        }
        
        # Handle optional fields
        if "datasets" in meta_grp:
            metadata_dict["datasets"] = [s.decode() for s in meta_grp["datasets"][:]]
        if "avdeepfake1m_filter" in meta_grp.attrs:
            metadata_dict["avdeepfake1m_filter"] = meta_grp.attrs["avdeepfake1m_filter"]
        if "created_at_filter" in meta_grp.attrs:
            val = meta_grp.attrs["created_at_filter"]
            metadata_dict["created_at_filter"] = val.decode() if isinstance(val, bytes) else val
        
        data["metadata"] = metadata_dict
        
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
            
            video_dict = {
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
            
            # Handle optional fields
            if "dataset" in vid_grp.attrs:
                video_dict["dataset"] = vid_grp.attrs["dataset"].decode() if isinstance(vid_grp.attrs["dataset"], bytes) else vid_grp.attrs["dataset"]
            if "created_at" in vid_grp.attrs:
                val = vid_grp.attrs["created_at"]
                video_dict["created_at"] = val.decode() if isinstance(val, bytes) else val
            
            data["videos"][video_id] = video_dict
    
    return data

