from .hubert_embedder import HubertEmbedder
from .ridnet_embedder import RidnetEmbedder
from .demucs_openl3_embedder import DemucsOpenl3Embedder
from .openl3_embedder import Openl3Embedder
from .senet_embedder import SenetEmbedder
from .voicefixer_openl3_embedder import VoicefixerOpenl3Embedder

VIDEO_EMBEDDERS = [
    # RidnetEmbedder(),
    SenetEmbedder(),
]

AUDIO_EMBEDDERS = [
    HubertEmbedder(),
    DemucsOpenl3Embedder(),
    Openl3Embedder(),
    VoicefixerOpenl3Embedder(),
]