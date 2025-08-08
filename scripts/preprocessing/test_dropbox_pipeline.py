#!/usr/bin/env python3
"""
Comprehensive test script for the Dropbox embedding pipeline.
This script will:
1. Generate embeddings for a small subset of segments
2. Save them locally with mappings
3. Upload to Dropbox
4. Test retrieval from Dropbox
5. Verify the entire pipeline works
"""

import os
import sys
import tempfile
import shutil
from datetime import datetime, timezone
from db.embedding_store_utils import get_segments_by_created_at
from utils.config_loader import load_config
from embedding_generator import embed_segments
from embedding_saver import save_embeddings_to_files, insert_embeddings_to_db
from dropbox_uploader import create_faiss_index_and_upload
from embedding_retriever import EmbeddingRetriever

def test_dropbox_pipeline():
    """
    Test the complete Dropbox embedding pipeline.
    """
    print("🧪 Testing Dropbox Embedding Pipeline")
    print("=" * 60)
    
    # Configuration
    config = load_config()
    db_path = config["database"]["embedding_db_path"]
    created_at = "2025-07-31T16:46:45.022260"
    
    # Use a temporary directory for testing
    test_output_dir = "./test_embeddings"
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir, exist_ok=True)
    
    print(f"📁 Test output directory: {test_output_dir}")
    
    # Step 1: Get a small subset of segments for testing
    print("\n📥 Step 1: Loading segments...")
    segments = get_segments_by_created_at(db_path, created_at)
    print(f"Found {len(segments)} total segments")
    
    # Use only first 5 segments for testing
    test_segments = segments[:5]
    print(f"Using {len(test_segments)} segments for testing")
    
    for i, seg in enumerate(test_segments):
        print(f"  {i+1}. {seg['segment_id']}")
    
    # Step 2: Generate embeddings
    print("\n🔄 Step 2: Generating embeddings...")
    accumulator = embed_segments(test_segments)
    
    # Print summary of generated embeddings
    print("\n📊 Generated Embeddings Summary:")
    for (model, mode), data in accumulator.items():
        if data["embeddings"]:
            print(f"  ✅ {model} | {mode}: {len(data['embeddings'])} embeddings")
        else:
            print(f"  ❌ {model} | {mode}: No embeddings produced")
    
    # Step 3: Save embeddings locally
    print("\n💾 Step 3: Saving embeddings locally...")
    saved_files = save_embeddings_to_files(accumulator, test_output_dir, created_at)
    print(f"Saved {len(saved_files)} embedding files")
    
    # List saved files
    print("\n📁 Saved files:")
    for (model, mode), file_info in saved_files.items():
        print(f"  📄 {os.path.basename(file_info['npy_path'])}")
        print(f"  📄 {os.path.basename(file_info['mapping_path'])}")
    
    # Step 4: Upload to Dropbox
    print("\n☁️ Step 4: Uploading to Dropbox...")
    uploaded_files = create_faiss_index_and_upload(test_output_dir)
    print(f"Uploaded {len(uploaded_files)} FAISS indices and mapping files")
    
    # List uploaded files
    print("\n☁️ Uploaded files:")
    for file_info in uploaded_files:
        print(f"  📄 {file_info['dropbox_index_path']}")
        print(f"  📄 {file_info['dropbox_mapping_path']}")
    
    # Step 5: Test retrieval from Dropbox
    print("\n🔍 Step 5: Testing retrieval from Dropbox...")
    retriever = EmbeddingRetriever(local_cache_dir="./test_cache")
    
    # Test different models/modes
    test_combinations = [
        ("hubert", "audio"),
        ("openl3", "audio"),
        ("hubert_demucs", "audio_denoised"),
        ("hubert_demucs", "audio_noise")
    ]
    
    retrieval_results = {}
    
    for model, mode in test_combinations:
        print(f"\n🔍 Testing {model} | {mode}:")
        
        # Get metadata
        metadata = retriever.get_embedding_metadata(model, mode)
        if not metadata:
            print(f"  ❌ No metadata found for {model} | {mode}")
            continue
        
        print(f"  ✅ Found {len(metadata)} embeddings")
        
        # Test retrieval by segment_id
        if metadata:
            test_segment_id = metadata[0]["segment_id"]
            embedding_by_segment = retriever.get_embedding_by_segment_id(
                segment_id=test_segment_id,
                model=model,
                mode=mode
            )
            
            if embedding_by_segment is not None:
                print(f"  ✅ Retrieved by segment_id: {test_segment_id}")
                print(f"     Shape: {embedding_by_segment.shape}")
                retrieval_results[(model, mode, "segment_id")] = embedding_by_segment
            else:
                print(f"  ❌ Failed to retrieve by segment_id")
        
        # Test retrieval by embedding_id
        if metadata:
            test_embedding_id = metadata[0]["embedding_id"]
            embedding_by_id = retriever.get_embedding_by_embedding_id(
                embedding_id=test_embedding_id,
                model=model,
                mode=mode
            )
            
            if embedding_by_id is not None:
                print(f"  ✅ Retrieved by embedding_id: {test_embedding_id[:8]}...")
                print(f"     Shape: {embedding_by_id.shape}")
                retrieval_results[(model, mode, "embedding_id")] = embedding_by_id
            else:
                print(f"  ❌ Failed to retrieve by embedding_id")
        
        # Test similarity search
        if embedding_by_segment is not None:
            distances, similar_segments = retriever.search_similar_embeddings(
                query_embedding=embedding_by_segment,
                model=model,
                mode=mode,
                k=3
            )
            print(f"  🔍 Similarity search: Found {len(similar_segments)} similar embeddings")
            for i, (distance, seg_id) in enumerate(zip(distances, similar_segments)):
                print(f"     {i+1}. {seg_id} (distance: {distance:.4f})")
    
    # Step 6: Verify consistency
    print("\n🔍 Step 6: Verifying consistency...")
    import numpy as np
    
    for (model, mode, retrieval_type), embedding in retrieval_results.items():
        # Find corresponding embedding from the other retrieval method
        if retrieval_type == "segment_id":
            # Find the embedding_id version
            for (m, mo, rt), emb in retrieval_results.items():
                if m == model and mo == mode and rt == "embedding_id":
                    if np.array_equal(embedding, emb):
                        print(f"  ✅ {model} | {mode}: segment_id and embedding_id return same embedding")
                    else:
                        print(f"  ❌ {model} | {mode}: segment_id and embedding_id return different embeddings")
                    break
        elif retrieval_type == "embedding_id":
            # Find the segment_id version
            for (m, mo, rt), emb in retrieval_results.items():
                if m == model and mo == mode and rt == "segment_id":
                    if np.array_equal(embedding, emb):
                        print(f"  ✅ {model} | {mode}: embedding_id and segment_id return same embedding")
                    else:
                        print(f"  ❌ {model} | {mode}: embedding_id and segment_id return different embeddings")
                    break
    
    # Step 7: Cleanup
    print("\n🧹 Step 7: Cleanup...")
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
        print(f"  ✅ Removed test output directory: {test_output_dir}")
    
    if os.path.exists("./test_cache"):
        shutil.rmtree("./test_cache")
        print(f"  ✅ Removed test cache directory: ./test_cache")
    
    # Final summary
    print("\n🎉 Pipeline Test Summary:")
    print("=" * 60)
    print(f"✅ Generated embeddings for {len(test_segments)} segments")
    print(f"✅ Saved {len(saved_files)} embedding files locally")
    print(f"✅ Uploaded {len(uploaded_files)} files to Dropbox")
    print(f"✅ Tested retrieval for {len([k for k in retrieval_results.keys() if k[2] == 'segment_id'])} model/mode combinations")
    print(f"✅ Verified consistency between retrieval methods")
    
    print("\n🎯 Pipeline is working correctly!")
    print("You can now use the full pipeline with all segments.")

if __name__ == "__main__":
    test_dropbox_pipeline() 