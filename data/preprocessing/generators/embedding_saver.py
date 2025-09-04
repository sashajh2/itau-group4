import os
import uuid
import numpy as np
import json
from datetime import datetime, timezone
from db.embedding_store_utils import insert_embedding

def save_embeddings_to_files(accumulator, output_dir, created_at):
    """
    Save embeddings to .npy files with appropriate naming and create comprehensive mappings.
    
    Args:
        accumulator: Dictionary with (model, mode) keys containing embeddings and segment_ids
        output_dir: Directory to save the .npy files
        created_at: Timestamp for the embedding generation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    for (model, mode, noise), data in accumulator.items():
        if not data["embeddings"]:
            continue

        embs = np.stack(data["embeddings"])
        seg_ids = data["segment_ids"]

        # Create appropriate filename based on model and mode
        if mode in ["audio_denoised", "audio_noise"]:
            # Extract denoiser name from model (e.g., "hubert_demucs" -> "demucs")
            denoiser_name = model.split("_")[-1] if "_" in model else "unknown"
            base = f"{model}_{mode}_{noise}_{denoiser_name}_{created_at}"
        else:
            base = f"{model}_{mode}_{noise}_{created_at}"
            
        npy_path = os.path.join(output_dir, f"{base}.npy")
        np.save(npy_path, embs)
        
        # Create comprehensive mapping file
        mapping_path = os.path.join(output_dir, f"{base}_mapping.json")
        
        # Generate embedding_ids for each embedding
        embedding_ids = [str(uuid.uuid4()) for _ in range(len(seg_ids))]
        
        mapping = {
            "model": model,
            "mode": mode,
            "created_at": created_at,
            "total_embeddings": len(seg_ids),
            "embedding_dimension": embs.shape[1],
            
            # Primary mappings
            "segment_to_index": {seg_id: idx for idx, seg_id in enumerate(seg_ids)},
            "index_to_segment": {idx: seg_id for idx, seg_id in enumerate(seg_ids)},
            
            # Embedding ID mappings
            "segment_to_embedding_id": {seg_id: emb_id for seg_id, emb_id in zip(seg_ids, embedding_ids)},
            "embedding_id_to_index": {emb_id: idx for idx, emb_id in enumerate(embedding_ids)},
            "index_to_embedding_id": {idx: emb_id for idx, emb_id in enumerate(embedding_ids)},
            
            # Metadata for each embedding
            "embeddings_metadata": [
                {
                    "index": idx,
                    "segment_id": seg_id,
                    "embedding_id": emb_id,
                    "model": model,
                    "mode": mode,
                    "noise": noise
                }
                for idx, (seg_id, emb_id) in enumerate(zip(seg_ids, embedding_ids))
            ]
        }
        
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        saved_files[(model, mode, noise)] = {
            "npy_path": npy_path,
            "mapping_path": mapping_path,
            "shape": embs.shape,
            "segment_count": len(seg_ids),
            "embedding_ids": embedding_ids
        }
        
        print(f"✅ Saved: {npy_path} (shape: {embs.shape}, segments: {len(seg_ids)})")
        print(f"✅ Saved comprehensive mapping: {mapping_path}")
    
    return saved_files

def insert_embeddings_to_db(accumulator, db_path, created_at, output_dir):
    """
    Insert embedding metadata into the database with proper embedding_ids.
    
    Args:
        accumulator: Dictionary with (model, mode) keys containing embeddings and segment_ids
        db_path: Path to the database
        created_at: Timestamp for the embedding generation
        output_dir: Directory containing the saved .npy files
    """
    now = datetime.now(timezone.utc).isoformat()

    for (model, mode, noise), data in accumulator.items():
        if not data["embeddings"]:
            continue

        # Create appropriate filename based on model and mode
        if mode in ["audio_denoised", "audio_noise"]:
            # Extract denoiser name from model (e.g., "hubert_demucs" -> "demucs")
            denoiser_name = model.split("_")[-1] if "_" in model else "unknown"
            base = f"{model}_{mode}_{noise}_{denoiser_name}_{created_at}"
        else:
            base = f"{model}_{mode}_{created_at}"
            
        npy_path = os.path.join(output_dir, f"{base}.npy")
        mapping_path = os.path.join(output_dir, f"{base}_mapping.json")
        
        # Load mapping to get embedding_ids
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        
        # Debug: Print what we're working with
        print(f"🔍 Debug {model}_{mode}:")
        print(f"  - Accumulator segment_ids count: {len(data['segment_ids'])}")
        print(f"  - Mapping index_to_embedding_id count: {len(mapping['index_to_embedding_id'])}")
        print(f"  - First few accumulator segment_ids: {data['segment_ids'][:3]}")
        print(f"  - First few mapping indices: {list(mapping['index_to_embedding_id'].keys())[:3]}")
        
        embedding_id_to_segment = {
            mapping["embedding_id_to_index"][emb_id]: seg_id 
            for seg_id, emb_id in mapping["segment_to_embedding_id"].items()
        }

        for idx, sid in enumerate(data["segment_ids"]):
            embedding_id = mapping["index_to_embedding_id"][str(idx)]
            
            embedding_dict = {
                "embedding_id": embedding_id,
                "segment_id": sid,
                "mode": mode,
                "noise": noise,
                "model_name": model,
                "embedding_type": "raw",
                "reducer_id": None,
                "contraster_id": None,
                "embedding_path": npy_path,
                "created_at": now
            }
            insert_embedding(db_path, embedding_dict)

    print("✅ Inserted embeddings into DB with proper embedding_ids.") 