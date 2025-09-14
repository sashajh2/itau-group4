#!/usr/bin/env python3
"""
Retrieve embeddings and labels for a specific model and version.

This script:
1. Retrieves all embeddings from different shards for a given model/version
2. Stitches them together into a single .npy file
3. Pairs them with corresponding labels from the segments table
4. Returns both embeddings and labels as easily loadable files

Usage:
    python scripts/retrieve_embeddings.py --model hubert --version 2025-09-12 --mode audio --noise none
"""

import argparse
import os
import sqlite3
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import tempfile
import shutil

from utils.config_loader import load_config
from dropbox_utils.dropbox_utils import download_file


def get_embedding_metadata(db_path: str, model_name: str, version: str, 
                          mode: Optional[str] = None, noise: Optional[str] = None,
                          denoiser_name: Optional[str] = None) -> List[Dict]:
    """
    Retrieve embedding metadata from the database for a specific model and version.
    
    Args:
        db_path: Path to the SQLite database
        model_name: Name of the model (e.g., 'hubert', 'openl3', 'senet')
        version: Version string (e.g., '2025-09-12')
        mode: Optional mode filter ('audio' or 'video')
        noise: Optional noise filter ('none', 'denoised', 'noisy')
        denoiser_name: Optional denoiser filter ('demucs', 'voicefixer', etc.)
    
    Returns:
        List of embedding metadata dictionaries
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Build query with optional filters
        query = """
            SELECT embedding_id, segment_id, mode, noise, model_name, denoiser_name,
                   shard_path, row_index, vector_dim, dtype, version
            FROM embeddings 
            WHERE model_name = ? AND version = ?
        """
        params = [model_name, version]
        
        if mode is not None:
            query += " AND mode = ?"
            params.append(mode)
        
        if noise is not None:
            query += " AND noise = ?"
            params.append(noise)
            
        if denoiser_name is not None:
            query += " AND denoiser_name = ?"
            params.append(denoiser_name)
        
        query += " ORDER BY shard_path, row_index"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]


def get_segment_labels(db_path: str, segment_ids: List[str], mode: str) -> Dict[str, int]:
    """
    Retrieve labels for segments based on the mode.
    
    Args:
        db_path: Path to the SQLite database
        segment_ids: List of segment IDs to get labels for
        mode: 'audio' or 'video' to determine which label to retrieve
    
    Returns:
        Dictionary mapping segment_id to label
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Create placeholders for the IN clause
        placeholders = ','.join(['?'] * len(segment_ids))
        
        if mode == 'audio':
            label_column = 'audio_label'
        elif mode == 'video':
            label_column = 'video_label'
        else:
            raise ValueError(f"Mode must be 'audio' or 'video', got: {mode}")
        
        query = f"""
            SELECT segment_id, {label_column}
            FROM segments 
            WHERE segment_id IN ({placeholders})
        """
        
        cursor.execute(query, segment_ids)
        rows = cursor.fetchall()
        
        return {segment_id: label for segment_id, label in rows}


def download_and_load_shard(shard_path: str, temp_dir: str) -> np.ndarray:
    """
    Download a shard file from Dropbox and load it as a numpy array.
    
    Args:
        shard_path: Dropbox path to the shard file
        temp_dir: Temporary directory to download to
    
    Returns:
        Numpy array containing the embeddings from the shard
    """
    # Create local filename
    local_filename = os.path.basename(shard_path)
    local_path = os.path.join(temp_dir, local_filename)
    
    # Download the file
    print(f"Downloading {shard_path}...")
    metadata = download_file(shard_path, local_path)
    
    if metadata is None:
        raise RuntimeError(f"Failed to download {shard_path}")
    
    # Load the numpy array
    embeddings = np.load(local_path)
    print(f"Loaded shard with shape {embeddings.shape}")
    
    return embeddings


def retrieve_embeddings_and_labels(
    model_name: str,
    version: str,
    mode: Optional[str] = None,
    noise: Optional[str] = None,
    denoiser_name: Optional[str] = None,
    output_dir: str = "./embeddings/retrieved"
) -> Tuple[str, str]:
    """
    Retrieve all embeddings and labels for a specific model and version.
    
    Args:
        model_name: Name of the model (e.g., 'hubert', 'openl3', 'senet')
        version: Version string (e.g., '2025-09-12')
        mode: Optional mode filter ('audio' or 'video')
        noise: Optional noise filter ('none', 'denoised', 'noisy')
        denoiser_name: Optional denoiser filter ('demucs', 'voicefixer', etc.)
        output_dir: Directory to save the output files
    
    Returns:
        Tuple of (embeddings_file_path, labels_file_path)
    """
    # Load configuration
    config = load_config()
    db_path = config["database"]["embedding_db_path"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Retrieving embeddings for model={model_name}, version={version}")
    if mode:
        print(f"  Mode filter: {mode}")
    if noise:
        print(f"  Noise filter: {noise}")
    if denoiser_name:
        print(f"  Denoiser filter: {denoiser_name}")
    
    # Get embedding metadata
    print("Querying database for embedding metadata...")
    embedding_metadata = get_embedding_metadata(
        db_path, model_name, version, mode, noise, denoiser_name
    )
    
    if not embedding_metadata:
        raise ValueError(f"No embeddings found for model={model_name}, version={version}")
    
    print(f"Found {len(embedding_metadata)} embeddings")
    
    # Group by shard path
    shard_groups = {}
    for emb in embedding_metadata:
        shard_path = emb['shard_path']
        if shard_path not in shard_groups:
            shard_groups[shard_path] = []
        shard_groups[shard_path].append(emb)
    
    print(f"Found {len(shard_groups)} unique shards")
    
    # Create temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Download and process each shard
        all_embeddings = []
        all_segment_ids = []
        all_labels = []
        
        for shard_path, embeddings_in_shard in shard_groups.items():
            print(f"\nProcessing shard: {os.path.basename(shard_path)}")
            print(f"  Contains {len(embeddings_in_shard)} embeddings")
            
            # Download and load the shard
            shard_embeddings = download_and_load_shard(shard_path, temp_dir)
            
            # Extract embeddings and segment IDs in the correct order
            for emb_meta in embeddings_in_shard:
                row_idx = emb_meta['row_index']
                segment_id = emb_meta['segment_id']
                
                # Extract the specific embedding
                embedding = shard_embeddings[row_idx]
                all_embeddings.append(embedding)
                all_segment_ids.append(segment_id)
        
        print(f"\nTotal embeddings collected: {len(all_embeddings)}")
        
        # Convert to numpy arrays
        all_embeddings = np.array(all_embeddings)
        print(f"Final embeddings shape: {all_embeddings.shape}")
        
        # Get labels for all segments
        print("Retrieving segment labels...")
        if mode:
            # Use the specified mode
            label_mode = mode
        else:
            # Try to infer mode from the first embedding
            label_mode = embedding_metadata[0]['mode']
        
        print(f"Using mode '{label_mode}' for label retrieval")
        segment_labels = get_segment_labels(db_path, all_segment_ids, label_mode)
        
        # Create labels array in the same order as embeddings
        all_labels = [segment_labels[seg_id] for seg_id in all_segment_ids]
        all_labels = np.array(all_labels)
        
        print(f"Labels shape: {all_labels.shape}")
        print(f"Label distribution: {np.bincount(all_labels)}")
    
    # Create output filenames
    filename_parts = [model_name, version]
    if mode:
        filename_parts.append(mode)
    if noise:
        filename_parts.append(noise)
    if denoiser_name:
        filename_parts.append(denoiser_name)
    
    base_filename = "_".join(filename_parts)
    embeddings_file = os.path.join(output_dir, f"{base_filename}_embeddings.npy")
    labels_file = os.path.join(output_dir, f"{base_filename}_labels.pkl")
    
    # Save embeddings
    print(f"\nSaving embeddings to: {embeddings_file}")
    np.save(embeddings_file, all_embeddings)
    
    # Save labels and metadata
    print(f"Saving labels to: {labels_file}")
    labels_data = {
        'labels': all_labels,
        'segment_ids': all_segment_ids,
        'model_name': model_name,
        'version': version,
        'mode': label_mode,
        'noise': noise,
        'denoiser_name': denoiser_name,
        'num_embeddings': len(all_embeddings),
        'embedding_dim': all_embeddings.shape[1] if len(all_embeddings.shape) > 1 else 0
    }
    
    with open(labels_file, 'wb') as f:
        pickle.dump(labels_data, f)
    
    print(f"\n✅ Successfully retrieved embeddings and labels!")
    print(f"   Embeddings: {embeddings_file}")
    print(f"   Labels: {labels_file}")
    print(f"   Shape: {all_embeddings.shape}")
    print(f"   Labels: {len(all_labels)} (0: {np.sum(all_labels == 0)}, 1: {np.sum(all_labels == 1)})")
    
    return embeddings_file, labels_file


def main():
    parser = argparse.ArgumentParser(description="Retrieve embeddings and labels for a specific model and version")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., hubert, openl3, senet)")
    parser.add_argument("--version", type=str, required=True, help="Version string (e.g., 2025-09-12)")
    parser.add_argument("--mode", type=str, choices=['audio', 'video'], help="Mode filter (optional)")
    parser.add_argument("--noise", type=str, choices=['none', 'denoised', 'noisy'], help="Noise filter (optional)")
    parser.add_argument("--denoiser", type=str, help="Denoiser filter (optional, e.g., demucs, voicefixer)")
    parser.add_argument("--output-dir", type=str, default="./embeddings/retrieved", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        embeddings_file, labels_file = retrieve_embeddings_and_labels(
            model_name=args.model,
            version=args.version,
            mode=args.mode,
            noise=args.noise,
            denoiser_name=args.denoiser,
            output_dir=args.output_dir
        )
        
        print(f"\n🎉 Files saved successfully!")
        print(f"To load the data:")
        print(f"  embeddings = np.load('{embeddings_file}')")
        print(f"  with open('{labels_file}', 'rb') as f:")
        print(f"      labels_data = pickle.load(f)")
        print(f"      labels = labels_data['labels']")
        print(f"      segment_ids = labels_data['segment_ids']")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
