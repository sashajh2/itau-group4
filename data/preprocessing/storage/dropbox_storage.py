import os
import faiss
import dropbox
import json
import tempfile
import numpy as np
from dropbox.dropbox_utils import get_client

def check_dropbox_file_exists(dropbox_path):
    """
    Check if a file exists in Dropbox.
    
    Args:
        dbx: Dropbox client
        dropbox_path: Path in Dropbox
        
    Returns:
        bool: True if file exists, False otherwise
    """
    try:
        get_client().files_get_metadata(dropbox_path)
        return True
    except dropbox.exceptions.ApiError as e:
        # Check if the error is "not found"
        if hasattr(e.error, 'is_not_found') and e.error.is_not_found():
            return False
        elif hasattr(e.error, 'get_lookup_error') and e.error.get_lookup_error().is_not_found():
            return False
        elif hasattr(e.error, 'get_lookup_error') and hasattr(e.error.get_lookup_error(), 'is_not_found') and e.error.get_lookup_error().is_not_found():
            return False
        else:
            # For now, let's just check if the error message contains "not_found"
            error_str = str(e.error)
            if "not_found" in error_str.lower():
                return False
            else:
                raise e

def download_and_merge_faiss_index(dropbox_index_path, new_embeddings, new_mapping):
    """
    Download existing FAISS index, merge with new embeddings, and return merged index and mapping.
    
    Args:
        dbx: Dropbox client
        dropbox_index_path: Path to existing index in Dropbox
        new_embeddings: New embeddings to append
        new_mapping: New mapping data to merge
        
    Returns:
        tuple: (merged_faiss_index, merged_mapping)
    """
    # Download existing index to temp file
    with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as temp_index_file:
        temp_index_path = temp_index_file.name
    
    try:
        # Download existing index
        with open(temp_index_path, "wb") as f:
            metadata, response = get_client().files_download(dropbox_index_path)
            f.write(response.content)
        
        # Load existing index
        existing_index = faiss.read_index(temp_index_path)
        existing_embeddings_count = existing_index.ntotal
        
        # Add new embeddings to existing index
        existing_index.add(new_embeddings)
        
        # Download and merge mapping
        dropbox_mapping_path = dropbox_index_path.replace('.index', '_mapping.json')
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_mapping_file:
            temp_mapping_path = temp_mapping_file.name
        
        try:
            # Download existing mapping
            with open(temp_mapping_path, "wb") as f:
                metadata, response = get_client().files_download(dropbox_mapping_path)
                f.write(response.content)
            
            # Load existing mapping
            with open(temp_mapping_path, 'r') as f:
                existing_mapping = json.load(f)
            
            # Merge mappings
            merged_mapping = merge_mappings(existing_mapping, new_mapping, existing_embeddings_count)
            
        except dropbox.exceptions.ApiError as e:
            if e.error.is_not_found():
                # No existing mapping, use new mapping with offset
                merged_mapping = offset_mapping(new_mapping, existing_embeddings_count)
            else:
                raise e
        finally:
            if os.path.exists(temp_mapping_path):
                os.unlink(temp_mapping_path)
        
        return existing_index, merged_mapping
        
    finally:
        if os.path.exists(temp_index_path):
            os.unlink(temp_index_path)

def merge_mappings(existing_mapping, new_mapping, existing_count):
    """
    Merge existing mapping with new mapping, adjusting indices for new embeddings.
    
    Args:
        existing_mapping: Existing mapping dictionary
        new_mapping: New mapping dictionary
        existing_count: Number of existing embeddings
        
    Returns:
        dict: Merged mapping
    """
    merged_mapping = existing_mapping.copy()
    
    # Merge segment_to_index
    for segment_id, index in new_mapping.get("segment_to_index", {}).items():
        merged_mapping["segment_to_index"][segment_id] = index + existing_count
    
    # Merge embedding_id_to_index
    for embedding_id, index in new_mapping.get("embedding_id_to_index", {}).items():
        merged_mapping["embedding_id_to_index"][embedding_id] = index + existing_count
    
    # Merge metadata
    if "metadata" in new_mapping:
        if "metadata" not in merged_mapping:
            merged_mapping["metadata"] = []
        merged_mapping["metadata"].extend(new_mapping["metadata"])
    
    return merged_mapping

def offset_mapping(new_mapping, offset):
    """
    Add offset to all indices in a mapping.
    
    Args:
        new_mapping: Mapping dictionary
        offset: Offset to add to all indices
        
    Returns:
        dict: Mapping with offset applied
    """
    offset_mapping = {}
    
    # Offset segment_to_index
    if "segment_to_index" in new_mapping:
        offset_mapping["segment_to_index"] = {
            segment_id: index + offset 
            for segment_id, index in new_mapping["segment_to_index"].items()
        }
    
    # Offset embedding_id_to_index
    if "embedding_id_to_index" in new_mapping:
        offset_mapping["embedding_id_to_index"] = {
            embedding_id: index + offset 
            for embedding_id, index in new_mapping["embedding_id_to_index"].items()
        }
    
    # Copy metadata without offset
    if "metadata" in new_mapping:
        offset_mapping["metadata"] = new_mapping["metadata"]
    
    return offset_mapping

def create_faiss_index_and_upload(output_dir, dropbox_base_path="/embedding_store/AVDeepfake1M/raw/"):
    """
    Create FAISS indices for all .npy files and upload them to Dropbox.
    If indices already exist, append new embeddings to them.
    Also upload the corresponding mapping files.
    
    Args:
        output_dir: Directory containing the .npy embedding files
        dropbox_base_path: Base path in Dropbox for uploading indices
    """
    uploaded_files = []
    
    for fname in os.listdir(output_dir):
        if fname.endswith(".npy"):
            npy_path = os.path.join(output_dir, fname)
            mapping_path = npy_path.replace(".npy", "_mapping.json")
            embs = np.load(npy_path)
            
            # Parse filename to extract model and mode
            # Examples:
            # hubert_audio_2025-07-31T16:46:45.022260.npy
            # hubert_demucs_audio_denoised_demucs_2025-07-31T16:46:45.022260.npy
            # hubert_demucs_audio_noise_demucs_2025-07-31T16:46:45.022260.npy
            
            # Remove timestamp and extension
            base_name = fname.replace(".npy", "")
            parts = base_name.split("_")
            
            # Find the mode by looking for specific patterns
            mode = None
            model_name = None
            
            # Check for complex modes first
            if "audio_denoised" in base_name:
                mode = "audio_denoised"
                # Find where audio_denoised starts
                mode_start = base_name.find("audio_denoised")
                model_name = base_name[:mode_start-1]  # Remove trailing underscore
            elif "audio_noise" in base_name:
                mode = "audio_noise"
                # Find where audio_noise starts
                mode_start = base_name.find("audio_noise")
                model_name = base_name[:mode_start-1]  # Remove trailing underscore
            elif "audio" in base_name and "audio_denoised" not in base_name and "audio_noise" not in base_name:
                mode = "audio"
                # Find where audio starts
                mode_start = base_name.find("audio")
                model_name = base_name[:mode_start-1]  # Remove trailing underscore
            elif "video" in base_name:
                mode = "video"
                # Find where video starts
                mode_start = base_name.find("video")
                model_name = base_name[:mode_start-1]  # Remove trailing underscore
            else:
                print(f"⚠️ Could not parse mode from filename: {fname}")
                continue
            
            print(f"  📝 Parsed: model='{model_name}', mode='{mode}' from '{fname}'")
            
            # Create Dropbox paths
            dropbox_index_path = f"{dropbox_base_path}{mode}/{model_name}.index"
            dropbox_mapping_path = f"{dropbox_base_path}{mode}/{model_name}_mapping.json"
            
            # Load mapping file
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    mapping_data = json.load(f)
            else:
                print(f"⚠️ Mapping file not found: {mapping_path}")
                mapping_data = {}
            
            # Check if index already exists in Dropbox
            index_exists = check_dropbox_file_exists(dropbox_index_path)
            
            if index_exists:
                print(f"📥 Found existing index, appending to: {dropbox_index_path}")
                # Download existing index and merge
                faiss_index, merged_mapping = download_and_merge_faiss_index(
                    dropbox_index_path, embs, mapping_data
                )
            else:
                print(f"🆕 Creating new index: {dropbox_index_path}")
                # Create new FAISS index
                faiss_index = faiss.IndexFlatL2(embs.shape[1])
                faiss_index.add(embs)
                merged_mapping = mapping_data
            
            # Save FAISS index locally temporarily
            faiss_path = npy_path.replace(".npy", ".faiss")
            faiss.write_index(faiss_index, faiss_path)
            print(f"✅ FAISS index written: {faiss_path}")
            
            # Upload FAISS index to Dropbox
            try:
                with open(faiss_path, "rb") as f:
                    get_client().files_upload(
                        f.read(), 
                        dropbox_index_path, 
                        mode=dropbox.files.WriteMode.overwrite
                    )
                print(f"☁️ Uploaded FAISS index to Dropbox: {dropbox_index_path}")
            except Exception as e:
                print(f"❌ Failed to upload FAISS index {dropbox_index_path}: {e}")
                continue
            
            # Upload mapping file to Dropbox
            try:
                # Save merged mapping to temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_mapping_file:
                    temp_mapping_path = temp_mapping_file.name
                    json.dump(merged_mapping, temp_mapping_file, indent=2)
                
                # Upload mapping file
                with open(temp_mapping_path, "rb") as f:
                    get_client().files_upload(
                        f.read(),
                        dropbox_mapping_path,
                        mode=dropbox.files.WriteMode.overwrite
                    )
                print(f"☁️ Uploaded mapping file to Dropbox: {dropbox_mapping_path}")
                
                # Clean up temp file
                os.unlink(temp_mapping_path)
                
            except Exception as e:
                print(f"❌ Failed to upload mapping file {dropbox_mapping_path}: {e}")
            
            uploaded_files.append({
                "local_index_path": faiss_path,
                "local_mapping_path": mapping_path,
                "dropbox_index_path": dropbox_index_path,
                "dropbox_mapping_path": dropbox_mapping_path,
                "model": model_name,
                "mode": mode,
                "shape": embs.shape,
                "appended": index_exists
            })
    
    print(f"✅ Uploaded {len(uploaded_files)} FAISS indices and mapping files to Dropbox")
    return uploaded_files 