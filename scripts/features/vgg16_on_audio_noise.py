import matplotlib as plt
import numpy as np
import cv2
from tqdm import tqdm

import librosa
import os
from moviepy import VideoFileClip
from audio_signals import compute_melspectrogram, compute_spectrogram

from retriever.embedders import denoisers


def load_audio_from_videos(base_dir, sr=44100):
    audio_clips = []
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".mp4"):
                path = os.path.join(root, file)
                clip = VideoFileClip(path)
                
                if clip.audio is None:
                    print(f"⚠️ Skipping {path}: no audio stream")
                    continue
                
                try:
                    # Try normal method
                    audio = clip.audio.to_soundarray(fps=sr)
                except Exception as e:
                    print(f"⚠️ to_soundarray failed on {path}, trying fallback... {e}")
                    frames = [frame for frame in clip.audio.iter_frames(fps=sr, dtype="float32")]
                    if not frames:
                        print(f"⚠️ No frames extracted from {path}")
                        continue
                    audio = np.vstack(frames)
                
                # Convert to mono if stereo
                if audio.ndim == 2 and audio.shape[1] == 2:
                    audio = np.mean(audio, axis=1)
                else:
                    audio = audio.squeeze()
                
                # Append (numpy_array, filename)
                audio_clips.append(audio)
    
    return audio_clips

def denoised_audio(audios, sr = 44100):
    demucs_denoiser = denoisers.DemucsDenoiser()
    audio_clips = []
    for audio in tqdm(audios, desc="Denoising audio"):
        audio_clips.append(demucs_denoiser.split_audio(audio)) # tuple of denoised, noise
    return audio_clips


def prepare_for_vgg16(S_db):
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    S_img = (S_norm * 255).astype(np.uint8)
    S_rgb = np.stack([S_img]*3, axis=-1)
    S_resized = cv2.resize(S_rgb, (224, 224))
    return S_resized

clips = load_audio_from_videos("/Users/jeffzhu/Desktop/itau-group4/data/temp_video_extracted/AV1M/extracted/train/lrs3")
print(denoised_audio(clips).shape)