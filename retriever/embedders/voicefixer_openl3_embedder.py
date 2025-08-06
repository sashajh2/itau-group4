import numpy as np
import tempfile
import os
import soundfile as sf
import librosa
import torch
from voicefixer import VoiceFixer
import openl3

class VoicefixerOpenl3Embedder:
    def __init__(self):
        self.model_name = "voicefixer_openl3"
        self.mode = "audio noise"
        self.vf_model = VoiceFixer()

    def _denoise_with_voicefixer(self, audio, sr=44100, mode=0):
        audio_normalized = np.clip(audio, -1.0, 1.0)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as input_file:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as output_file:
                try:
                    sf.write(input_file.name, audio_normalized, sr)

                    self.vf_model.restore(
                        input=input_file.name,
                        output=output_file.name,
                        cuda=torch.cuda.is_available(),
                        mode=mode
                    )

                    if not os.path.exists(output_file.name):
                        raise RuntimeError("❌ VoiceFixer output file not created")

                    enhanced_audio, enhanced_sr = sf.read(output_file.name)

                    if enhanced_sr != sr:
                        enhanced_audio = librosa.resample(enhanced_audio, orig_sr=enhanced_sr, target_sr=sr)

                    if enhanced_audio.ndim == 2:
                        enhanced_audio = np.mean(enhanced_audio, axis=1)

                    return enhanced_audio.astype(np.float32), audio_normalized

                finally:
                    for temp_path in [input_file.name, output_file.name]:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)

    def _get_residual_noise(self, original, enhanced):
        min_len = min(len(original), len(enhanced))
        return original[:min_len] - enhanced[:min_len]

    def get_audio_noise(self, audio_array: np.ndarray) -> np.ndarray:
        enhanced, normalized = self._denoise_with_voicefixer(audio_array)
        return self._get_residual_noise(normalized, enhanced)

    def embed(self, audio_array: np.ndarray, sr: int) -> np.ndarray:
        clip_duration = audio_array.shape[0] / sr
        emb, ts = openl3.get_audio_embedding(audio_array, sr, input_repr="mel256", hop_size=clip_duration, center=False, embedding_size=512)
        return emb.mean(axis=0)
