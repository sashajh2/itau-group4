import numpy as np
import torch

class Args:
    def __init__(self):
        self.n_feats = 64
        self.rgb_range = 255
        self.reduction = 16

class RidnetEmbedder:
    def __init__(self):
        self.model_name = "ridnet_pca_128"
        self.mode = "video noise"
        self.model = self.load_model()  

    def load_model(self):
        from models.pretrained.ridnet.ridnet import RIDNET
        args = Args()
        model = RIDNET(args)
        model.load_state_dict(torch.load("models/pretrained/ridnet/ridnet.pt"))
        model.eval()
        return model
    
    def get_video_noise(self, video: str) -> str:
        pass

    def embed(self, video: str, num_components: int = 128) -> np.ndarray:
        pass