class DemucsOpenl3Embedder:
    def __init__(self):
        pass

    def embed(self, text: str) -> np.ndarray:
        pass

    @property
    def model_name(self) -> str:
        return "demucs_openl3"
    @property
    def mode(self):
        return "audio noise"