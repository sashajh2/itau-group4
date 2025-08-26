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
        print(f"🔍 Input audio: shape={audio_array.shape}, dtype={audio_array.dtype}, sr={sr}")
        
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

        # Ensure audio is float32 and normalize
        audio_array = audio_array.astype(np.float32)
        
        # Resample to target SR expected by HuBERT (16kHz)
        target_sr = self.sr
        if sr != target_sr:
            print(f"🔄 Resampling audio from {sr}Hz to {target_sr}Hz (length: {len(audio_array)} -> {int(len(audio_array) * target_sr / sr)})")
            # Use librosa.resample with proper parameters for better quality
            audio_array = librosa.resample(
                y=audio_array, 
                orig_sr=sr, 
                target_sr=target_sr,
                res_type='kaiser_best'  # Higher quality resampling
            )
            sr = target_sr
            print(f"✅ Resampling complete. New length: {len(audio_array)}, New SR: {sr}")

        # Ensure audio is not too long (HuBERT has limits)
        max_length = int(30 * sr)  # 30 seconds max
        if len(audio_array) > max_length:
            audio_array = audio_array[:max_length]

        # Convert to float32 tensor, add batch dim
        inputs = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        print(f"🎯 Input tensor shape: {inputs.shape}, dtype: {inputs.dtype}")

        with torch.no_grad():
            outputs = self.model(inputs)

        # Average over time dimension → shape (1, dim)
        hidden_states = outputs.last_hidden_state
        embedding = hidden_states.mean(dim=1).squeeze().numpy()
        print(f"🎉 HuBERT embedding generated: shape {embedding.shape}")
        return embedding
