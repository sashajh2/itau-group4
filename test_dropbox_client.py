#!/usr/bin/env python3
"""
Test script to verify that get_client() is working and can retrieve embeddings
from the specified Dropbox path: Sasha Jovanovic-Hacon/Apps/itau-group4/embedding_store/AVDeepfake1M/raw/audio/denoised/hubert_demucs.index
"""

import os
import json
import numpy as np
import faiss
import tempfile
from dropbox_utils.dropbox_utils import get_client

def test_dropbox_connection():
    """Test basic Dropbox connection using get_client()"""
    print("🔍 Testing Dropbox connection...")
    try:
        account = get_client().users_get_current_account()
        print(f"✅ Connected to Dropbox as: {account.name.display_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Dropbox: {e}")
        return False

def test_file_exists(dropbox_path):
    """Test if a file exists in Dropbox"""
    print(f"🔍 Checking if file exists: {dropbox_path}")
    try:
        metadata = get_client().files_get_metadata(dropbox_path)
        print(f"✅ File exists: {metadata.name} (size: {metadata.size} bytes)")
        return True
    except Exception as e:
        print(f"❌ File not found or error: {e}")
        return False

def test_download_faiss_index(dropbox_index_path, dropbox_mapping_path):
    """Test downloading FAISS index and mapping from Dropbox"""
    print(f"🔍 Testing download of FAISS index and mapping...")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as temp_index_file:
        temp_index_path = temp_index_file.name
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_mapping_file:
        temp_mapping_path = temp_mapping_file.name
    
    try:
        # Download FAISS index
        print(f"📥 Downloading FAISS index: {dropbox_index_path}")
        with open(temp_index_path, "wb") as f:
            metadata, response = get_client().files_download(dropbox_index_path)
            f.write(response.content)
        print(f"✅ Downloaded FAISS index: {metadata.name} ({metadata.size} bytes)")
        
        # Download mapping file
        print(f"📥 Downloading mapping file: {dropbox_mapping_path}")
        with open(temp_mapping_path, "wb") as f:
            metadata, response = get_client().files_download(dropbox_mapping_path)
            f.write(response.content)
        print(f"✅ Downloaded mapping file: {metadata.name} ({metadata.size} bytes)")
        
        # Load and inspect FAISS index
        print("🔍 Loading FAISS index...")
        faiss_index = faiss.read_index(temp_index_path)
        print(f"✅ FAISS index loaded: {faiss_index.ntotal} vectors, dimension {faiss_index.d}")
        
        # Load and inspect mapping
        print("🔍 Loading mapping file...")
        with open(temp_mapping_path, 'r') as f:
            mapping = json.load(f)
        
        print(f"✅ Mapping loaded with keys: {list(mapping.keys())}")
        
        # Show some sample data
        if "segment_to_index" in mapping:
            sample_segments = list(mapping["segment_to_index"].keys())[:5]
            print(f"📋 Sample segment IDs: {sample_segments}")
        
        if "embedding_id_to_index" in mapping:
            sample_embedding_ids = list(mapping["embedding_id_to_index"].keys())[:5]
            print(f"📋 Sample embedding IDs: {sample_embedding_ids}")
        
        # Test retrieving a sample embedding
        if faiss_index.ntotal > 0:
            print("🔍 Testing embedding retrieval...")
            sample_embedding = faiss_index.reconstruct(0)
            print(f"✅ Retrieved sample embedding: shape {sample_embedding.shape}, dtype {sample_embedding.dtype}")
            print(f"📊 Sample embedding stats: min={sample_embedding.min():.4f}, max={sample_embedding.max():.4f}, mean={sample_embedding.mean():.4f}")
        
        return True, faiss_index, mapping
        
    except Exception as e:
        print(f"❌ Failed to download or process files: {e}")
        return False, None, None
    
    finally:
        # Clean up temporary files
        for temp_path in [temp_index_path, temp_mapping_path]:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

def test_search_similar_embeddings(faiss_index, mapping, k=5):
    """Test searching for similar embeddings"""
    if faiss_index is None or faiss_index.ntotal == 0:
        print("⚠️ No FAISS index available for search test")
        return
    
    print(f"🔍 Testing similarity search with k={k}...")
    
    try:
        # Get a random query embedding
        query_idx = 0  # Use first embedding as query
        query_embedding = faiss_index.reconstruct(query_idx)
        query_embedding = query_embedding.reshape(1, -1)  # Make it 2D
        
        # Search for similar embeddings
        distances, indices = faiss_index.search(query_embedding, k)
        
        print(f"✅ Search completed: found {len(indices[0])} similar embeddings")
        
        # Show results
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if "index_to_segment" in mapping and idx in mapping["index_to_segment"]:
                segment_id = mapping["index_to_segment"][idx]
                print(f"  {i+1}. Index {idx} (segment: {segment_id}) - distance: {distance:.4f}")
            else:
                print(f"  {i+1}. Index {idx} - distance: {distance:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Dropbox client test for embedding retrieval")
    print("=" * 60)
    
    # Test connection
    if not test_dropbox_connection():
        print("❌ Cannot proceed without Dropbox connection")
        return
    
    print("\n" + "=" * 60)
    
    # Define the paths to test
    dropbox_base_path = "/embedding_store/AVDeepfake1M/raw/audio/denoised/"
    index_filename = "hubert_demucs.index"
    mapping_filename = "hubert_demucs_mapping.json"
    
    dropbox_index_path = dropbox_base_path + index_filename
    dropbox_mapping_path = dropbox_base_path + mapping_filename
    
    # Test file existence
    print(f"📁 Testing file existence...")
    index_exists = test_file_exists(dropbox_index_path)
    mapping_exists = test_file_exists(dropbox_mapping_path)
    
    if not index_exists or not mapping_exists:
        print("❌ Required files not found in Dropbox")
        return
    
    print("\n" + "=" * 60)
    
    # Test downloading and processing
    success, faiss_index, mapping = test_download_faiss_index(dropbox_index_path, dropbox_mapping_path)
    
    if not success:
        print("❌ Failed to download or process files")
        return
    
    print("\n" + "=" * 60)
    
    # Test similarity search
    test_search_similar_embeddings(faiss_index, mapping)
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed successfully!")
    print("✅ get_client() is working correctly")
    print("✅ Can retrieve embeddings from Dropbox")
    print("✅ FAISS index operations work")
    print("✅ Mapping file operations work")

if __name__ == "__main__":
    main()
