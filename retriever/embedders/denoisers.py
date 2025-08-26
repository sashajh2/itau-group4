import numpy as np
import torch
import tempfile
import os
import soundfile as sf
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model
from voicefixer import VoiceFixer

class DemucsDenoiser:
    def __init__(self):
        self.model = self.load_demucs_model()

    def load_demucs_model(self):
        model = get_model(name='htdemucs')
        model.eval()
        return model

    def denoise(self, audio_array: np.ndarray) -> np.ndarray:
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1)  # [2, T]
        elif audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.repeat(2, 1)

        wav = audio_tensor.unsqueeze(0)  # [1, 2, T]

        with torch.no_grad():
            sources = apply_model(self.model, wav, split=True, overlap=0.25, progress=False)

        sources = sources.numpy()[0]
        denoised = sources[3]  # vocals
        return denoised.flatten()
    
    def get_noise(self, audio_array: np.ndarray) -> np.ndarray:
        denoised = self.denoise(audio_array)
        min_len = min(len(audio_array), len(denoised))
        residual = audio_array[:min_len] - denoised[:min_len]
        return residual

    def split_audio(self, audio_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Split audio into denoised and noise tracks.
        
        Args:
            audio_array: Input audio array
            
        Returns:
            tuple: (denoised_audio, noise_audio)
        """
        denoised = self.denoise(audio_array)
        min_len = min(len(audio_array), len(denoised))
        denoised = denoised[:min_len]
        original = audio_array[:min_len]
        noise = original - denoised
        return denoised, noise

class VoiceFixerDenoiser:
    def __init__(self):
        self.vf_model = VoiceFixer()

    def denoise(self, audio, sr=44100, mode=0):
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

                    return enhanced_audio.astype(np.float32)

                finally:
                    for temp_path in [input_file.name, output_file.name]:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)

    def get_noise(self, audio_array: np.ndarray) -> np.ndarray:
        enhanced = self.denoise(audio_array)
        min_len = min(len(audio_array), len(enhanced))
        return audio_array[:min_len] - enhanced[:min_len]

    def split_audio(self, audio_array: np.ndarray, sr=44100, mode=0) -> tuple[np.ndarray, np.ndarray]:
        """
        Split audio into denoised and noise tracks.
        
        Args:
            audio_array: Input audio array
            sr: Sample rate (default: 44100)
            mode: VoiceFixer mode (default: 0)
            
        Returns:
            tuple: (denoised_audio, noise_audio)
        """
        denoised = self.denoise(audio_array, sr, mode)
        min_len = min(len(audio_array), len(denoised))
        denoised = denoised[:min_len]
        original = audio_array[:min_len]
        noise = original - denoised
        return denoised, noise 