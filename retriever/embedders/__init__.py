from .hubert_embedder import HubertEmbedder
from .ridnet_pca_embedder import RidnetEmbedder
from .openl3_embedder import Openl3Embedder
from .senet_embedder import SenetEmbedder
from .denoisers import DemucsDenoiser, VoiceFixerDenoiser

# Denoisers
DENOISERS = {
    "demucs": DemucsDenoiser(),
    "voicefixer": VoiceFixerDenoiser(),
}

VIDEO_EMBEDDERS = [
    # RidnetEmbedder(),
    SenetEmbedder(),
]

AUDIO_EMBEDDERS = [
    # Regular audio embedders only
    HubertEmbedder(mode="audio"),
    Openl3Embedder(mode="audio"),
]