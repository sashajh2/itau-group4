class RidnetPca128Embedder:
    def __init__(self):
        pass

    def embed(self, text: str) -> np.ndarray:
        pass

    @property
    def model_name(self) -> str:
        return "ridnet_pca_128"
    @property
    def mode(self):
        return "video noise"