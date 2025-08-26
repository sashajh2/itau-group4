import numpy as np
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import openl3

class DemucsOpenl3Embedder:
    def __init__(self):
        self.model_name = "demucs_openl3"
        self.mode = "audio noise"
        self.model = self.load_demucs_model()

    def load_demucs_model(self):
        model = get_model(name='htdemucs')
        model.eval()
        return model

    def denoise_with_demucs(self, audio_array: np.ndarray) -> np.ndarray:

        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1)  # [2, T]
        elif audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.repeat(2, 1)

        wav = audio_tensor.unsqueeze(0)  # [1, 2, T]

        with torch.no_grad():
            sources = apply_model(self.model, wav, split=True, overlap=0.25, progress=False)

        sources = sources.numpy()[0]
        denoised = sources[3]  # vocals
        return denoised.flatten()
    
    def get_audio_noise(self, audio_array: np.ndarray) -> np.ndarray:
        denoised = self.denoise_with_demucs(audio_array)
        min_len = min(len(audio_array), len(denoised))
        residual = audio_array[:min_len] - denoised[:min_len]
        return residual

    def embed(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        clip_duration = audio_array.shape[0] / sr
        emb, ts = openl3.get_audio_embedding(audio_array, sr, input_repr="mel256", hop_size=clip_duration, center=False, embedding_size=512)
        return emb.mean(axis=0)
