import numpy as np

class RidnetPca128Embedder:
    def __init__(self):
        self.model_name = "ridnet_pca_128"
        self.mode = "video noise"
    
    def get_video_noise(self, video: str) -> str:
        pass

    def embed(self, video: str) -> np.ndarray:
        pass