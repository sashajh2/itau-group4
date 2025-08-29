# Accepts a query (text, vector, etc.), 
# performs vector search in FAISS, 
# retrieves metadata from SQLite, 
# and returns relevant segments. 
# 
# Can be turned into an API or used in a notebook.

import numpy as np
import pickle
import json
import sqlite3
import faiss
from pathlib import Path
from typing import Tuple, Dict, List
from dropbox.dropbox_utils import download_file
from utils.config_loader import load_config

def retrieve_hubert_embeddings_and_labels(
    created_at: str = "2025-08-21T13:51:04.162022+00:00",
    output_dir: str = "./embeddings/audio/hubert",
    dropbox_embedding_path: str = "/embedding_store/AVDeepfake1M/raw/audio/hubert.index",
    dropbox_mapping_path: str = "/embedding_store/AVDeepfake1M/raw/audio/hubert_mapping.json",
    local_temp_dir: str = "./temp_embeddings"
) -> Tuple[np.ndarray, List[int]]:
    """
    Retrieve Hubert embeddings from FAISS index and corresponding labels from database.
    
    Args:
        created_at: Timestamp of the embedding index to retrieve
        output_dir: Directory to save the output .npy and .pkl files
        dropbox_embedding_path: Path to the Hubert FAISS index in Dropbox
        dropbox_mapping_path: Path to the Hubert mapping JSON in Dropbox
        local_temp_dir: Local directory for temporary files
        
    Returns:
        Tuple of (embeddings_array, labels_list)
    """
    
    # Create output and temp directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(local_temp_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Retrieving Hubert embeddings for {created_at}")
    
    # Download files from Dropbox
    print("📥 Downloading files from Dropbox...")
    
    local_index_path = f"{local_temp_dir}/hubert.index"
    local_mapping_path = f"{local_temp_dir}/hubert_mapping.json"
    
    # Download FAISS index
    if not download_file(dropbox_embedding_path, local_index_path):
        raise RuntimeError("Failed to download Hubert FAISS index from Dropbox")
    
    # Download mapping JSON
    if not download_file(dropbox_mapping_path, local_mapping_path):
        raise RuntimeError("Failed to download Hubert mapping JSON from Dropbox")
    
    print("✅ Files downloaded successfully")
    
    # Load the mapping JSON
    print("📋 Loading segment-to-index mapping...")
    with open(local_mapping_path, 'r') as f:
        mapping_data = json.load(f)
    
    segment_to_index = mapping_data.get('segment_to_index', {})
    total_embeddings = mapping_data.get('total_embeddings', 0)
    embedding_dim = mapping_data.get('embedding_dim', 768)
    
    print(f"📊 Found {total_embeddings} embeddings with dimension {embedding_dim}")
    print(f"📝 Found {len(segment_to_index)} segment mappings")
    
    # Load the FAISS index
    print("🔍 Loading FAISS index...")
    index = faiss.read_index(local_index_path)
    print(f"📊 FAISS index loaded: {index.ntotal} vectors")
    
    # Retrieve all embeddings
    print("🔄 Retrieving all embeddings from FAISS...")
    embeddings = index.reconstruct_n(0, total_embeddings)
    print(f"✅ Retrieved embeddings shape: {embeddings.shape}")
    
    # Connect to SQLite database
    print("🗄️ Connecting to SQLite database...")
    config = load_config()
    db_path = config.get('database', {}).get('path', './db/embeddings.sqlite3')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get segments table schema to understand the structure
    cursor.execute("PRAGMA table_info(segments)")
    columns = cursor.fetchall()
    print(f"📋 Segments table columns: {[col[1] for col in columns]}")
    
    # Query database for each segment to get audio labels
    print("🏷️ Retrieving audio labels from database...")
    labels = []
    segment_ids = []
    
    for segment_id, index_num in segment_to_index.items():
        try:
            # Query the segments table for this segment_id
            cursor.execute(
                "SELECT audio_label FROM segments WHERE segment_id = ? AND created_at = ?",
                (segment_id, created_at)
            )
            result = cursor.fetchone()
            
            if result:
                audio_label = result[0]
                labels.append(audio_label)
                segment_ids.append(segment_id)
            else:
                print(f"⚠️ No database entry found for segment: {segment_id}")
                # Use a default label (you might want to adjust this)
                labels.append(-1)  # Default label for missing entries
                segment_ids.append(segment_id)
                
        except Exception as e:
            print(f"❌ Error querying segment {segment_id}: {e}")
            labels.append(-1)  # Default label for errors
            segment_ids.append(segment_id)
    
    conn.close()
    
    print(f"🏷️ Retrieved {len(labels)} labels")
    print(f"📊 Label distribution: {np.bincount(np.array(labels))}")
    
    # Save embeddings as .npy
    embeddings_output_path = f"{output_dir}/hubert_embeddings_{created_at.replace(':', '-').replace('+', '_')}.npy"
    print(f"💾 Saving embeddings to {embeddings_output_path}")
    np.save(embeddings_output_path, embeddings)
    
    # Save labels as .pkl
    labels_output_path = f"{output_dir}/hubert_labels_{created_at.replace(':', '-').replace('+', '_')}.pkl"
    print(f"💾 Saving labels to {labels_output_path}")
    
    labels_data = {
        'labels': labels,
        'segment_ids': segment_ids,
        'created_at': created_at,
        'total_embeddings': total_embeddings,
        'embedding_dim': embedding_dim
    }
    
    with open(labels_output_path, 'wb') as f:
        pickle.dump(labels_data, f)
    
    # Clean up temporary files
    print("🧹 Cleaning up temporary files...")
    Path(local_index_path).unlink(missing_ok=True)
    Path(local_mapping_path).unlink(missing_ok=True)
    
    print(f"🎉 Successfully saved {total_embeddings} Hubert embeddings and labels!")
    print(f"📁 Embeddings: {embeddings_output_path}")
    print(f"📁 Labels: {labels_output_path}")
    
    return embeddings, labels

def main():
    """Example usage of the retrieval function"""
    try:
        embeddings, labels = retrieve_hubert_embeddings_and_labels()
        
        print(f"\n📊 Final Summary:")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Labels count: {len(labels)}")
        print(f"   Unique labels: {np.unique(labels)}")
        
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()