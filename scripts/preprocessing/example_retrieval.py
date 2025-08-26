#!/usr/bin/env python3
"""
Example script demonstrating how to retrieve embeddings from FAISS indices stored in Dropbox.
"""

from embedding_retriever import EmbeddingRetriever
import numpy as np

def main():
    """
    Demonstrate embedding retrieval functionality.
    """
    print("🔍 Embedding Retrieval Example")
    print("=" * 50)
    
    # Initialize retriever
    retriever = EmbeddingRetriever()
    
    # Example segment IDs (you would get these from your database)
    example_segments = [
        "3p3Svl9VFYU/00006/00006_p1_2340",
        "3p3Svl9VFYU/00006/00006_p1_2350",
        "3p3Svl9VFYU/00006/00006_p1_2360"
    ]
    
    # Example 1: Retrieve a single embedding
    print("\n📥 Example 1: Retrieve single embedding")
    print("-" * 40)
    
    embedding = retriever.get_embedding_by_segment_id(
        segment_id=example_segments[0],
        model="hubert",
        mode="audio"
    )
    
    if embedding is not None:
        print(f"✅ Retrieved embedding for {example_segments[0]}")
        print(f"   Shape: {embedding.shape}")
        print(f"   Type: {embedding.dtype}")
        print(f"   Range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    else:
        print(f"❌ Failed to retrieve embedding for {example_segments[0]}")
    
    # Example 2: Retrieve multiple embeddings
    print("\n📥 Example 2: Retrieve multiple embeddings")
    print("-" * 40)
    
    embeddings = retriever.get_embeddings_by_segment_ids(
        segment_ids=example_segments,
        model="hubert",
        mode="audio"
    )
    
    print(f"✅ Retrieved {len(embeddings)} embeddings:")
    for seg_id, emb in embeddings.items():
        print(f"   {seg_id}: shape {emb.shape}")
    
    # Example 3: Search for similar embeddings
    print("\n🔍 Example 3: Search for similar embeddings")
    print("-" * 40)
    
    if embedding is not None:
        distances, similar_segments = retriever.search_similar_embeddings(
            query_embedding=embedding,
            model="hubert",
            mode="audio",
            k=5
        )
        
        print(f"✅ Found {len(similar_segments)} similar embeddings:")
        for i, (distance, seg_id) in enumerate(zip(distances, similar_segments)):
            print(f"   {i+1}. {seg_id} (distance: {distance:.4f})")
    
    # Example 4: Compare different models/modes
    print("\n🔄 Example 4: Compare different models/modes")
    print("-" * 40)
    
    models_modes = [
        ("hubert", "audio"),
        ("hubert_demucs", "audio_denoised"),
        ("hubert_demucs", "audio_noise"),
        ("openl3", "audio")
    ]
    
    for model, mode in models_modes:
        print(f"\n   Testing {model} | {mode}:")
        emb = retriever.get_embedding_by_segment_id(
            segment_id=example_segments[0],
            model=model,
            mode=mode
        )
        
        if emb is not None:
            print(f"     ✅ Retrieved embedding (shape: {emb.shape})")
        else:
            print(f"     ❌ Failed to retrieve embedding")

if __name__ == "__main__":
    main() 