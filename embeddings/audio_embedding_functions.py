import numpy as np  # type: ignore
import librosa  # type: ignore

def embedding_mfcc(audio_clip, sr=16000):
    """
    MFCC embedding extraction.
    """
    y = audio_clip.to_soundarray(fps=sr).mean(axis=1)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1)

# Wav2Vec2
from transformers import Wav2Vec2Processor, Wav2Vec2Model  # type: ignore
import torch  # type: ignore

def load_wav2vec2():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()
    return processor, model

def embedding_wav2vec(audio_clip, sr=16000, processor=None, model=None):
    """
    Wav2Vec2 embedding extraction.
    """
    if processor is None or model is None:
        processor, model = load_wav2vec2()
    y = audio_clip.to_soundarray(fps=sr).mean(axis=1)
    input_values = processor(y, sampling_rate=sr, return_tensors="pt").input_values
    with torch.no_grad():
        outputs = model(input_values)
    hidden_states = outputs.last_hidden_state
    return hidden_states.mean(dim=1).squeeze().numpy()

# HuBERT
from transformers import HubertModel  # type: ignore

def load_hubert():
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    model.eval()
    return model

def embedding_hubert(audio_clip, sr=16000, model=None):
    """
    HuBERT embedding extraction.
    """
    if model is None:
        model = load_hubert()
    y = audio_clip.to_soundarray(fps=sr).mean(axis=1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        outputs = model(y)
    hidden_states = outputs.last_hidden_state
    return hidden_states.mean(dim=1).squeeze().numpy()

# VGGish
import tensorflow_hub as hub  # type: ignore

def load_vggish():
    return hub.load("https://www.kaggle.com/models/google/vggish/TensorFlow2/vggish/1")

def embedding_vggish(audio_clip, sr=16000, model=None):
    """
    VGGish embedding extraction.
    """
    if model is None:
        model = load_vggish()
    y = audio_clip.to_soundarray(fps=sr).mean(axis=1).astype(np.float32)
    target_len = int(sr * 0.96)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    embeddings = model(y).numpy()
    return embeddings.mean(axis=0)

# OpenL3
import openl3  # type: ignore

def embedding_openl3(audio_clip, sr=48000):
    """
    OpenL3 embedding extraction.
    """
    y = audio_clip.to_soundarray(fps=sr).mean(axis=1)
    emb, ts = openl3.get_audio_embedding(y, sr, input_repr="mel256", content_type="env", embedding_size=512)
    return emb.mean(axis=0) 