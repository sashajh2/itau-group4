from .hubert_embedder import HubertEmbedder
from .ridnet_pca_embedder import RidnetEmbedder
from .demucs_openl3_embedder import DemucsOpenl3Embedder
from .openl3_embedder import Openl3Embedder
from .senet_embedder import SenetEmbedder
from .voicefixer_openl3_embedder import VoicefixerOpenl3Embedder
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
    # Regular audio embedders
    HubertEmbedder(mode="audio"),
    Openl3Embedder(mode="audio"),
    
    # Denoised audio embedders
    HubertEmbedder(mode="audio_denoised"),
    Openl3Embedder(mode="audio_denoised"),
    
    # Noise audio embedders  
    HubertEmbedder(mode="audio_noise"),
    Openl3Embedder(mode="audio_noise"),
]