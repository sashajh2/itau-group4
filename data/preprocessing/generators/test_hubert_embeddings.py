#!/usr/bin/env python3
"""
Test script to generate Hubert embeddings for a small subset of segments.
This follows the embed_segments pattern but only processes Hubert audio embeddings.
"""

import numpy as np
from db.embedding_store_utils import get_segments_by_created_at
from utils.config_loader import load_config
from retriever.embedders.hubert_embedder import HubertEmbedder
from moviepy import VideoFileClip
from utils.embedding_utils import get_audio_array


def test_hubert_embeddings(created_at: str, limit: int = 5) -> dict:
    """
    Generate Hubert embeddings for a limited number of segments.
    
    Args:
        created_at: Timestamp to get segments for
        limit: Maximum number of segments to process
        
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
    
    # Limit to first N segments for testing
    test_segments = segments[:limit]
    print(f"🧪 Testing with {len(test_segments)} segments")
    
    # Initialize Hubert embedder
    hubert_embedder = HubertEmbedder(mode="audio")
    
    # Store results
    results = {
        "embeddings": [],
        "segment_ids": [],
        "video_paths": [],
        "start_times": [],
        "durations": []
    }
    
    for i, segment in enumerate(test_segments):
        segment_id = segment["segment_id"]
        filepath = segment["video_path"]
        start_time = segment["start_time"]
        duration = segment["duration"]
        
        print(f"\n📹 Processing segment {i+1}/{len(test_segments)}: {segment_id}")
        print(f"   File: {filepath}")
        print(f"   Time: {start_time}s to {start_time + duration}s")
        
        try:
            # Load video and extract audio segment
            video = VideoFileClip(filepath)
            video = video.subclipped(start_time, start_time + duration)
            audio = video.audio
            audio_array = get_audio_array(audio, 16000)
            
            print(f"   Audio: shape={audio_array.shape}")
            
            # Generate Hubert embedding
            print(f"   🧠 Generating Hubert embedding...")
            embedding = hubert_embedder.embed(audio_array)
            print(f"   ✅ Hubert embedding: shape={embedding.shape}")
            
            # Store results
            results["embeddings"].append(embedding)
            results["segment_ids"].append(segment_id)
            results["video_paths"].append(filepath)
            results["start_times"].append(start_time)
            results["durations"].append(duration)
            
        except Exception as e:
            print(f"   ❌ Error processing segment {segment_id}: {e}")
            continue
    
    print(f"\n🎉 Successfully processed {len(results['embeddings'])} segments")
    return results


def compare_with_local_embeddings(hubert_results: dict, local_npy_path: str = "embeddings/audio/hubert/unified_hubert_embeddings.npy"):
    """
    Compare generated Hubert embeddings with local .npy file.
    
    Args:
        hubert_results: Results from test_hubert_embeddings
        local_npy_path: Path to local Hubert embeddings
    """
    try:
        print(f"\n🔍 Loading local embeddings from: {local_npy_path}")
        local_embs = np.load(local_npy_path)
        print(f"Local embeddings: shape={local_embs.shape}")
        
        if len(hubert_results["embeddings"]) == 0:
            print("❌ No Hubert embeddings to compare")
            return
        
        # Convert to numpy array
        generated_embs = np.array(hubert_results["embeddings"])
        print(f"Generated embeddings: shape={generated_embs.shape}")
        
        # Check dimensions match
        if generated_embs.shape[1] != local_embs.shape[1]:
            print(f"❌ Dimension mismatch: generated={generated_embs.shape[1]}, local={local_embs.shape[1]}")
            return
        
        # Compare first few embeddings
        print(f"\n📊 Comparison Results:")
        for i in range(len(generated_embs)):
            generated_emb = generated_embs[i]
            segment_id = hubert_results["segment_ids"][i]
            
            # Find closest match in local embeddings
            distances = np.linalg.norm(local_embs - generated_emb, axis=1)
            closest_idx = np.argmin(distances)
            closest_distance = distances[closest_idx]
            
            print(f"  Segment {segment_id}:")
            print(f"    Closest local embedding: index {closest_idx}, distance: {closest_distance:.6f}")
            print(f"    Generated embedding norm: {np.linalg.norm(generated_emb):.6f}")
            print(f"    Local embedding norm: {np.linalg.norm(local_embs[closest_idx]):.6f}")
        
        # Overall statistics
        print(f"\n📈 Overall Statistics:")
        print(f"  Generated embeddings mean norm: {np.mean([np.linalg.norm(emb) for emb in generated_embs]):.6f}")
        print(f"  Local embeddings mean norm: {np.mean([np.linalg.norm(emb) for emb in local_embs]):.6f}")
        
    except Exception as e:
        print(f"❌ Error comparing with local embeddings: {e}")


def main():
    """Main test function."""
    # Test with the specific created_at timestamp
    created_at = "2025-08-21T13:51:04.162022+00:00"
    
    print("🧪 Testing Hubert Embedding Generation")
    print("=" * 50)
    
    # Generate embeddings for 5 segments
    results = test_hubert_embeddings(created_at, limit=5)
    
    if results:
        # Compare with local embeddings
        compare_with_local_embeddings(results)
        
        # Save test results for inspection
        test_output = {
            "embeddings": np.array(results["embeddings"]),
            "segment_ids": results["segment_ids"],
            "video_paths": results["video_paths"],
            "start_times": results["start_times"],
            "durations": results["durations"]
        }
        
        output_path = f"test_hubert_embeddings_{created_at.replace(':', '-')}.npz"
        np.savez(output_path, **test_output)
        print(f"\n💾 Test results saved to: {output_path}")
        
        print(f"\n✅ Test completed successfully!")
        print(f"   Processed {len(results['embeddings'])} segments")
        print(f"   All embeddings have shape: {results['embeddings'][0].shape if results['embeddings'] else 'N/A'}")
    else:
        print("❌ Test failed - no embeddings generated")


if __name__ == "__main__":
    main()
