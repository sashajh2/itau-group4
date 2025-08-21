# retriever/embedders/hubert_embedder.py

import numpy as np
import torch
import librosa  # type: ignore
from transformers import HubertModel
from utils.config_loader import load_config

class HubertEmbedder:
    def __init__(self, mode="audio", sr=16000):
        config = load_config()
        self.hf_token = config["huggingface"]["token"]

        self.model_name = "hubert"
        self.mode = mode  # "audio", "audio_noise", "audio_denoised"
        # HuBERT expects 16 kHz inputs. Keep target SR configurable but default to 16k.
        self.sr = sr

        # Load once
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960", token=self.hf_token)
        self.model.eval()

    def embed(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        """Return HuBERT embedding for a single audio array.

        Ensures mono and resamples to self.sr (default 16 kHz) before inference.
        """
        # Downmix to mono robustly
        if isinstance(audio_array, np.ndarray) and audio_array.ndim > 1:
            # Heuristic: if one dimension is small (<= 8), treat that as channels and average across it
            if audio_array.shape[-1] <= 8:
                audio_array = audio_array.mean(axis=-1)
            elif audio_array.shape[0] <= 8:
                audio_array = audio_array.mean(axis=0)
            else:
                # Fallback: average across last axis
                audio_array = audio_array.mean(axis=-1)

        # Resample to target SR expected by HuBERT
        target_sr = self.sr
        if sr != target_sr:
            audio_array = librosa.resample(y=audio_array.astype(np.float32), orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Convert to float32 tensor, add batch dim
        inputs = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(inputs)

        # Average over time dimension → shape (1, dim)
        hidden_states = outputs.last_hidden_state
        return hidden_states.mean(dim=1).squeeze().numpy()
