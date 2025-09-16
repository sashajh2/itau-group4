import matplotlib as plt
import numpy as np
import cv2
from tqdm import tqdm
import json

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
                
                # Determine label from JSON
                json_path = path.replace(".mp4", ".json")
                label = 0  # default: real
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        meta = json.load(f)
                        modify_type = meta.get("modify_type", "")
                        if modify_type in ["both_modified", "audio_modified"]:
                            label = 1  # fake
                
                # Append (numpy_array, label)
                audio_clips.append((audio,label))
                clip.close()
    
    return audio_clips

def denoised_and_noise_audio(audios, sr = 44100):
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


def convert_to_spectrogram_for_vgg16(audios, sr=44100):
    images = []
    for audio in tqdm(audios, desc="Converting audios to spectrograms"):
        S_db = compute_spectrogram(audio, sr)
        images.append(prepare_for_vgg16(S_db))
    return images

def convert_to_melspectrogram_for_vgg16(audios, sr=44100):
    images = []
    for audio in tqdm(audios, desc="Converting audios to spectrograms"):
        S_db = compute_melspectrogram(audio, sr)
        images.append(prepare_for_vgg16(S_db))
    return images

def save_images(images, out_dir, prefix="spectrogram"):
    """
    Save a list of spectrogram/mel-spectrogram images to a directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        out_path = os.path.join(out_dir, f"{prefix}_{i}.png")
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # OpenCV saves in BGR


clips = load_audio_from_videos("/Users/jeffzhu/Desktop/itau-group4/data/temp_video_extracted/AV1M/extracted/train/lrs3")
denoised, noise = denoised_and_noise_audio(clips[0][0])
denoised_spectrogram = convert_to_spectrogram_for_vgg16(denoised)
denoised_melspectrogram = convert_to_melspectrogram_for_vgg16(denoised)
noise_spectrogram = convert_to_spectrogram_for_vgg16(noise)
noise_melspectrogram = convert_to_melspectrogram_for_vgg16(noise)

# Saves images
save_images(denoised_spectrogram, "outputs/denoised/spectrogram", prefix="denoised_spec")
save_images(denoised_melspectrogram, "outputs/denoised/melspectrogram", prefix="denoised_mel")
save_images(noise_spectrogram, "outputs/noise/spectrogram", prefix="noise_spec")
save_images(noise_melspectrogram, "outputs/noise/melspectrogram", prefix="noise_mel")

