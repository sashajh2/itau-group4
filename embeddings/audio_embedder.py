from .embedder import Embedder
from typing import Optional, Any, Callable
import numpy as np  # type: ignore
from .audio_embedding_functions import (
    embedding_mfcc,
    embedding_wav2vec,
    embedding_hubert,
    embedding_vggish,
    embedding_openl3,
)

class AudioEmbedder(Embedder):
    """
    Handles audio embedding extraction, saving, and loading for audio models.
    """
    def __init__(self, model_name: str, checkpoint_path: Optional[str] = None, device: str = 'cpu'):
        super().__init__(model_name, checkpoint_path, device)

    def get_embedding_fn(self, model_name: str) -> Callable:
        """
        Returns the correct embedding function for the given model_name.
        """
        if model_name.lower() == 'mfcc':
            return embedding_mfcc
        elif model_name.lower() == 'wav2vec2':
            return embedding_wav2vec
        elif model_name.lower() == 'hubert':
            return embedding_hubert
        elif model_name.lower() == 'vggish':
            return embedding_vggish
        elif model_name.lower() == 'openl3':
            return embedding_openl3
        else:
            raise ValueError(f"Unknown audio model: {model_name}")
