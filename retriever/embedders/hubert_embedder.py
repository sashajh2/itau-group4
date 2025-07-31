class HubertEmbedder:
    def __init__(self):
        pass

    def embed(self, text: str) -> np.ndarray:
        pass

    @property
    def model_name(self) -> str:
        return "hubert"
    @property
    def mode(self):
        return "audio"