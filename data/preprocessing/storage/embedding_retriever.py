import os
import json
import numpy as np
import faiss
import dropbox
from utils.config_loader import load_config

class EmbeddingRetriever:
    """
    Utility class to retrieve specific embeddings from FAISS indices stored in Dropbox.
    Supports both segment_id and embedding_id based retrieval.
    """
    
    def __init__(self, local_cache_dir="./embeddings/cache"):
        self.config = load_config()
        self.access_token = self.config["dropbox"]["access_token"]
        self.dbx = dropbox.Dropbox(self.access_token)
        self.local_cache_dir = local_cache_dir
        os.makedirs(local_cache_dir, exist_ok=True)
    
    def download_faiss_index(self, model, mode, dropbox_base_path="/embedding_store/AVDeepfake1M/raw/"):
        """
        Download FAISS index and mapping file from Dropbox.
        
        Args:
            model: Model name (e.g., "hubert", "hubert_demucs")
            mode: Mode (e.g., "audio", "audio_denoised", "audio_noise", "video")
            dropbox_base_path: Base path in Dropbox
            
        Returns:
            tuple: (faiss_index, mapping_dict, local_index_path)
        """
        dropbox_index_path = f"{dropbox_base_path}{mode}/{model}.index"
        dropbox_mapping_path = f"{dropbox_base_path}{mode}/{model}_mapping.json"
        
        # Local cache paths
        local_index_path = os.path.join(self.local_cache_dir, f"{model}_{mode}.faiss")
        local_mapping_path = os.path.join(self.local_cache_dir, f"{model}_{mode}_mapping.json")
        
        # Download FAISS index
        try:
            with open(local_index_path, "wb") as f:
                metadata, response = self.dbx.files_download(dropbox_index_path)
                f.write(response.content)
            print(f"✅ Downloaded FAISS index: {dropbox_index_path}")
        except Exception as e:
            print(f"❌ Failed to download FAISS index: {e}")
            return None, None, None
        
        # Download mapping file
        try:
            with open(local_mapping_path, "wb") as f:
                metadata, response = self.dbx.files_download(dropbox_mapping_path)
                f.write(response.content)
            print(f"✅ Downloaded mapping file: {dropbox_mapping_path}")
        except Exception as e:
            print(f"❌ Failed to download mapping file: {e}")
            return None, None, None
        
        # Load FAISS index
        faiss_index = faiss.read_index(local_index_path)
        
        # Load mapping
        with open(local_mapping_path, 'r') as f:
            mapping = json.load(f)
        
        return faiss_index, mapping, local_index_path
    
    def get_embedding_by_segment_id(self, segment_id, model, mode):
        """
        Retrieve a specific embedding by segment_id.
        
        Args:
            segment_id: The segment ID to retrieve
            model: Model name
            mode: Mode name
            
        Returns:
            numpy.ndarray: The embedding vector, or None if not found
        """
        faiss_index, mapping, _ = self.download_faiss_index(model, mode)
        
        if faiss_index is None or mapping is None:
            return None
        
        # Get index position for this segment
        segment_to_index = mapping["segment_to_index"]
        if segment_id not in segment_to_index:
            print(f"❌ Segment {segment_id} not found in index")
            return None
        
        index_pos = segment_to_index[segment_id]
        
        # Retrieve the embedding
        embedding = faiss_index.reconstruct(index_pos)
        return embedding
    
    def get_embedding_by_embedding_id(self, embedding_id, model, mode):
        """
        Retrieve a specific embedding by embedding_id.
        
        Args:
            embedding_id: The embedding ID to retrieve
            model: Model name
            mode: Mode name
            
        Returns:
            numpy.ndarray: The embedding vector, or None if not found
        """
        faiss_index, mapping, _ = self.download_faiss_index(model, mode)
        
        if faiss_index is None or mapping is None:
            return None
        
        # Get index position for this embedding_id
        embedding_id_to_index = mapping["embedding_id_to_index"]
        if embedding_id not in embedding_id_to_index:
            print(f"❌ Embedding ID {embedding_id} not found in index")
            return None
        
        index_pos = embedding_id_to_index[embedding_id]
        
        # Retrieve the embedding
        embedding = faiss_index.reconstruct(index_pos)
        return embedding
    
    def get_embeddings_by_segment_ids(self, segment_ids, model, mode):
        """
        Retrieve multiple embeddings by segment_ids.
        
        Args:
            segment_ids: List of segment IDs to retrieve
            model: Model name
            mode: Mode name
            
        Returns:
            dict: {segment_id: embedding_vector}
        """
        faiss_index, mapping, _ = self.download_faiss_index(model, mode)
        
        if faiss_index is None or mapping is None:
            return {}
        
        segment_to_index = mapping["segment_to_index"]
        results = {}
        
        for segment_id in segment_ids:
            if segment_id in segment_to_index:
                index_pos = segment_to_index[segment_id]
                embedding = faiss_index.reconstruct(index_pos)
                results[segment_id] = embedding
            else:
                print(f"⚠️ Segment {segment_id} not found in index")
        
        return results
    
    def get_embeddings_by_embedding_ids(self, embedding_ids, model, mode):
        """
        Retrieve multiple embeddings by embedding_ids.
        
        Args:
            embedding_ids: List of embedding IDs to retrieve
            model: Model name
            mode: Mode name
            
        Returns:
            dict: {embedding_id: embedding_vector}
        """
        faiss_index, mapping, _ = self.download_faiss_index(model, mode)
        
        if faiss_index is None or mapping is None:
            return {}
        
        embedding_id_to_index = mapping["embedding_id_to_index"]
        results = {}
        
        for embedding_id in embedding_ids:
            if embedding_id in embedding_id_to_index:
                index_pos = embedding_id_to_index[embedding_id]
                embedding = faiss_index.reconstruct(index_pos)
                results[embedding_id] = embedding
            else:
                print(f"⚠️ Embedding ID {embedding_id} not found in index")
        
        return results
    
    def get_embedding_metadata(self, model, mode):
        """
        Get metadata for all embeddings in an index.
        
        Args:
            model: Model name
            mode: Mode name
            
        Returns:
            list: List of embedding metadata dictionaries
        """
        _, mapping, _ = self.download_faiss_index(model, mode)
        
        if mapping is None:
            return []
        
        return mapping.get("embeddings_metadata", [])
    
    def search_similar_embeddings(self, query_embedding, model, mode, k=10, return_metadata=False):
        """
        Search for similar embeddings using FAISS.
        
        Args:
            query_embedding: Query embedding vector
            model: Model name
            mode: Mode name
            k: Number of similar embeddings to return
            return_metadata: Whether to return full metadata or just segment_ids
            
        Returns:
            tuple: (distances, segment_ids) or (distances, metadata_list) if return_metadata=True
        """
        faiss_index, mapping, _ = self.download_faiss_index(model, mode)
        
        if faiss_index is None or mapping is None:
            return [], []
        
        # Reshape query to 2D if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = faiss_index.search(query_embedding, k)
        
        if return_metadata:
            # Return full metadata for each result
            metadata_list = []
            for idx in indices[0]:
                if idx in mapping["index_to_embedding_id"]:
                    embedding_id = mapping["index_to_embedding_id"][idx]
                    segment_id = mapping["index_to_segment"][idx]
                    metadata_list.append({
                        "index": int(idx),
                        "segment_id": segment_id,
                        "embedding_id": embedding_id,
                        "model": model,
                        "mode": mode
                    })
            return distances[0], metadata_list
        else:
            # Convert indices to segment_ids
            index_to_segment = mapping["index_to_segment"]
            segment_ids = [index_to_segment.get(idx, f"unknown_{idx}") for idx in indices[0]]
            return distances[0], segment_ids

# Example usage functions
def retrieve_embedding_example():
    """
    Example of how to use the EmbeddingRetriever with both segment_id and embedding_id.
    """
    retriever = EmbeddingRetriever()
    
    # Get metadata to see available embeddings
    metadata = retriever.get_embedding_metadata("hubert", "audio")
    if metadata:
        example_segment = metadata[0]["segment_id"]
        example_embedding_id = metadata[0]["embedding_id"]
        
        print(f"Example segment_id: {example_segment}")
        print(f"Example embedding_id: {example_embedding_id}")
        
        # Get embedding by segment_id
        embedding_by_segment = retriever.get_embedding_by_segment_id(
            segment_id=example_segment,
            model="hubert",
            mode="audio"
        )
        
        # Get embedding by embedding_id
        embedding_by_id = retriever.get_embedding_by_embedding_id(
            embedding_id=example_embedding_id,
            model="hubert",
            mode="audio"
        )
        
        if embedding_by_segment is not None and embedding_by_id is not None:
            # Verify they're the same
            if np.array_equal(embedding_by_segment, embedding_by_id):
                print("✅ Both retrieval methods return the same embedding!")
            else:
                print("❌ Retrieval methods return different embeddings")
    
    # Search with metadata
    if embedding_by_segment is not None:
        distances, metadata_list = retriever.search_similar_embeddings(
            query_embedding=embedding_by_segment,
            model="hubert",
            mode="audio",
            k=5,
            return_metadata=True
        )
        
        print(f"\nTop 5 similar embeddings with metadata:")
        for i, (distance, meta) in enumerate(zip(distances, metadata_list)):
            print(f"  {i+1}. {meta['segment_id']} (distance: {distance:.4f})")

if __name__ == "__main__":
    retrieve_embedding_example() 