from .embedder import Embedder
from typing import Optional, Any
import numpy as np  # type: ignore

class AudioEmbedder(Embedder):
    """
    Handles audio embedding extraction, saving, and loading for audio models.
    """
    def __init__(self, model_name: str, checkpoint_path: Optional[str] = None, device: str = 'cpu'):
        super().__init__(model_name, checkpoint_path, device)

    def embedding_fn(self, audio_clip: Any) -> np.ndarray:
        """
        Model-specific embedding extraction for audio. To be implemented.
        """
        # TODO: Implement actual embedding logic
        return np.zeros(512)

    def embed_dataset(self, loader, batch_size: int = 32) -> Tuple[np.ndarray, List[Any]]:
        """
        Run embedding extraction over a dataset loader.
        Returns: (embeddings, metadata)
        """
        embeddings = []
        metadata = []
        for batch in loader:
            batch_embeds, batch_meta = self.extract_embeddings(batch)
            embeddings.append(batch_embeds)
            metadata.extend(batch_meta)
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings, metadata

    def extract_embeddings(self, batch: Any) -> Tuple[np.ndarray, List[Any]]:
        """
        Extract embeddings for a batch. Fill in model-specific logic here.
        Returns: (embeddings, metadata)
        """
        # TODO: Implement embedding extraction logic
        # Example: audio = batch['audio']
        # embeddings = self.model(audio)
        # meta = batch['meta']
        # return embeddings.cpu().numpy(), meta
        return np.zeros((len(batch), 512)), [{} for _ in range(len(batch))]

    def save_embeddings(self, embeddings: np.ndarray, metadata: List[Any], out_dir: str):
        """
        Save embeddings and metadata to disk.
        """
        np.save(f"{out_dir}/embeddings.npy", embeddings)
        import pickle
        with open(f"{out_dir}/meta.pkl", "wb") as f:
            pickle.dump(metadata, f)

    def load_embeddings(self, in_dir: str) -> Tuple[np.ndarray, List[Any]]:
        """
        Load embeddings and metadata from disk.
        """
        embeddings = np.load(f"{in_dir}/embeddings.npy")
        import pickle
        with open(f"{in_dir}/meta.pkl", "rb") as f:
            metadata = pickle.load(f)
        return embeddings, metadata
