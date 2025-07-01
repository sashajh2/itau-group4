import numpy as np  # type: ignore
from typing import Any, List, Tuple

class FaissIndexer:
    """
    Handles FAISS index creation, saving, loading, and searching.
    """
    def __init__(self, dim: int):
        try:
            import faiss  # type: ignore
        except ImportError:
            raise ImportError("faiss is required for FaissIndexer")
        self.faiss = faiss
        self.index = self.faiss.IndexFlatL2(dim)

    def build_index(self, embeddings: np.ndarray):
        self.index.add(embeddings.astype(np.float32))

    def save_index(self, out_path: str):
        self.faiss.write_index(self.index, out_path)

    def load_index(self, in_path: str):
        self.index = self.faiss.read_index(in_path)

    def search(self, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the index for the k nearest neighbors of the query.
        Returns: (distances, indices)
        """
        return self.index.search(query.astype(np.float32), k)
