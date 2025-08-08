import os
import faiss
import dropbox
from utils.config_loader import load_config
import numpy as np

def create_faiss_index_and_upload(output_dir, dropbox_base_path="/embedding_store/AVDeepfake1M/raw/"):
    """
    Create FAISS indices for all .npy files and upload them to Dropbox.
    Also upload the corresponding mapping files.
    
    Args:
        output_dir: Directory containing the .npy embedding files
        dropbox_base_path: Base path in Dropbox for uploading indices
    """
    config = load_config()
    access_token = config["dropbox"]["access_token"]
    dbx = dropbox.Dropbox(access_token)

    uploaded_files = []
    
    for fname in os.listdir(output_dir):
        if fname.endswith(".npy"):
            npy_path = os.path.join(output_dir, fname)
            mapping_path = npy_path.replace(".npy", "_mapping.json")
            embs = np.load(npy_path)
            
            # Create FAISS index
            index = faiss.IndexFlatL2(embs.shape[1])
            index.add(embs)
            
            # Save FAISS index locally
            faiss_path = npy_path.replace(".npy", ".faiss")
            faiss.write_index(index, faiss_path)
            print(f"✅ FAISS index written: {faiss_path}")
            
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
            
            # Upload FAISS index to Dropbox
            try:
                with open(faiss_path, "rb") as f:
                    dbx.files_upload(
                        f.read(), 
                        dropbox_index_path, 
                        mode=dropbox.files.WriteMode.overwrite
                    )
                print(f"☁️ Uploaded FAISS index to Dropbox: {dropbox_index_path}")
            except Exception as e:
                print(f"❌ Failed to upload FAISS index {dropbox_index_path}: {e}")
                continue
            
            # Upload mapping file to Dropbox
            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, "rb") as f:
                        dbx.files_upload(
                            f.read(),
                            dropbox_mapping_path,
                            mode=dropbox.files.WriteMode.overwrite
                        )
                    print(f"☁️ Uploaded mapping file to Dropbox: {dropbox_mapping_path}")
                except Exception as e:
                    print(f"❌ Failed to upload mapping file {dropbox_mapping_path}: {e}")
            else:
                print(f"⚠️ Mapping file not found: {mapping_path}")
            
            uploaded_files.append({
                "local_index_path": faiss_path,
                "local_mapping_path": mapping_path,
                "dropbox_index_path": dropbox_index_path,
                "dropbox_mapping_path": dropbox_mapping_path,
                "model": model_name,
                "mode": mode,
                "shape": embs.shape
            })
    
    print(f"✅ Uploaded {len(uploaded_files)} FAISS indices and mapping files to Dropbox")
    return uploaded_files 