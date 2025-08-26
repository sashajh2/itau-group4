import numpy as np
import openl3

class Openl3Embedder:
    def __init__(self, mode="audio"):
        self.model_name = "openl3"
        self.mode = mode  # "audio", "audio_noise", "audio_denoised"

    def embed(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        clip_duration = audio_array.shape[0] / sr
        emb, ts = openl3.get_audio_embedding(audio_array, sr, input_repr="mel256", hop_size=clip_duration, center=False, embedding_size=512)
        return emb.mean(axis=0)