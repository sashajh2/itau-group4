# retriever/embedders/hubert_embedder.py

import numpy as np
import torch
from transformers import HubertModel
from utils.config_loader import load_config

class HubertEmbedder:
    def __init__(self, sr=44100):
        config = load_config()
        self.hf_token = config["huggingface"]["token"]

        self.model_name = "hubert"
        self.mode = "audio"
        self.sr = sr

        # Load once
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960", token=self.hf_token)
        self.model.eval()

    def embed(self, audio_array, sr) -> np.ndarray:
        inputs = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(inputs)

        # Average over time dimension → shape (1, dim)
        hidden_states = outputs.last_hidden_state
        return hidden_states.mean(dim=1).squeeze().numpy()
