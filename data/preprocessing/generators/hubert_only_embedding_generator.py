#!/usr/bin/env python3
"""
Hubert-only embedding generator that follows the embed_segments pattern.
This generates Hubert embeddings for all segments with a given created_at timestamp.
"""

import numpy as np
from db.embedding_store_utils import get_segments_by_created_at
from utils.config_loader import load_config
from retriever.embedders.hubert_embedder import HubertEmbedder
from moviepy import VideoFileClip
from utils.embedding_utils import get_audio_array


def embed_segments_hubert_only(segments):
    """
    Generate Hubert embeddings for all segments.
    
    Args:
        segments: List of segment dictionaries
        
    Returns:
        Dictionary with Hubert embeddings and segment_ids
    """
    # Initialize Hubert embedder
    hubert_embedder = HubertEmbedder(mode="audio")
    
    # Store results
    results = {
        "embeddings": [],
        "segment_ids": []
    }
    
    total_segments = len(segments)
    
    for i, segment in enumerate(segments):
        if i % 100 == 0:
            print(f"Processing segment {i+1}/{total_segments}: {segment['segment_id']}")
            
        segment_id = segment["segment_id"]
        filepath = segment["video_path"]
        start_time = segment["start_time"]
        duration = segment["duration"]

        try:
            # Load video and extract audio segment
            video = VideoFileClip(filepath)
            video = video.subclipped(start_time, start_time + duration)
            audio = video.audio
            sr = audio.fps
            audio_array = get_audio_array(audio, sr)
            
            # Generate Hubert embedding
            embedding = hubert_embedder.embed(audio_array, sr)
            
            # Store results
            results["embeddings"].append(embedding)
            results["segment_ids"].append(segment_id)
            
        except Exception as e:
            print(f"❌ Error processing segment {segment_id}: {e}")
            continue
    
    print(f"✅ Successfully processed {len(results['embeddings'])} segments")
    return results


def generate_hubert_embeddings_for_created_at(created_at: str) -> dict:
    """
    Generate Hubert embeddings for all segments with the given created_at.
    
    Args:
        created_at: Timestamp to get segments for
        
    Returns:
        Dictionary with embeddings and segment_ids
    """
    config = load_config()
    db_path = config["database"]["embedding_db_path"]
    
    print(f"🔍 Getting segments for {created_at}")
    segments = get_segments_by_created_at(db_path, created_at)
    print(f"Found {len(segments)} segments")
    
    if len(segments) == 0:
        print("❌ No segments found for the specified created_at timestamp")
        return {}
    
    print(f"🧠 Generating Hubert embeddings for {len(segments)} segments...")
    results = embed_segments_hubert_only(segments)
    
    return results


def save_hubert_embeddings(results: dict, output_dir: str, created_at: str):
    """
    Save Hubert embeddings to files.
    
    Args:
        results: Dictionary with embeddings and segment_ids
        output_dir: Directory to save embeddings
        created_at: Timestamp for the embeddings
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not results or not results["embeddings"]:
        print("❌ No embeddings to save")
        return
    
    # Convert to numpy arrays
    embeddings = np.array(results["embeddings"])
    segment_ids = np.array(results["segment_ids"])
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, f"hubert_embeddings_{created_at.replace(':', '-')}.npy")
    np.save(embeddings_path, embeddings)
    print(f"💾 Saved embeddings to: {embeddings_path}")
    
    # Save segment IDs
    segment_ids_path = os.path.join(output_dir, f"hubert_segment_ids_{created_at.replace(':', '-')}.npy")
    np.save(segment_ids_path, segment_ids)
    print(f"💾 Saved segment IDs to: {segment_ids_path}")
    
    # Save metadata
    metadata = {
        "created_at": created_at,
        "num_segments": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "embeddings_file": os.path.basename(embeddings_path),
        "segment_ids_file": os.path.basename(segment_ids_path)
    }
    
    import json
    metadata_path = os.path.join(output_dir, f"hubert_metadata_{created_at.replace(':', '-')}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata to: {metadata_path}")
    
    return embeddings_path, segment_ids_path, metadata_path


def main():
    """Main function to generate Hubert embeddings for a specific created_at."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Hubert embeddings for segments with a specific created_at")
    parser.add_argument("--created-at", type=str, required=True, help="ISO8601 created_at partition to process")
    parser.add_argument("--output-dir", type=str, default="./embeddings/generated", help="Directory to save embeddings")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Hubert embedding generation for {args.created_at}")
    
    try:
        # Generate embeddings
        results = generate_hubert_embeddings_for_created_at(args.created_at)
        
        if results and results["embeddings"]:
            # Save embeddings
            embeddings_path, segment_ids_path, metadata_path = save_hubert_embeddings(
                results, args.output_dir, args.created_at
            )
            
            print(f"\n🎉 Hubert embedding generation completed successfully!")
            print(f"   Processed {len(results['embeddings'])} segments")
            print(f"   Embedding dimension: {results['embeddings'][0].shape}")
            print(f"   Output directory: {args.output_dir}")
        else:
            print("❌ No embeddings were generated")
            return 1
            
    except Exception as e:
        print(f"❌ Error during Hubert embedding generation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())