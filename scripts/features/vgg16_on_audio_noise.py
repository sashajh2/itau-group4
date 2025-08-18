import matplotlib as plt
import numpy as np
import cv2

import librosa
import os
from moviepy import AudioFileClip
from audio_signals import compute_melspectrogram, compute_spectrogram




def load_audio_from_videos(base_dir, sr=44100):
    audio_clips = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".mp4"):
                path = os.path.join(root, file)
                try:
                    y, _ = librosa.load(path, sr=sr)  # extracts audio only
                    audio_clips.append((y, path))
                except Exception as e:
                    print(f"Skipping {path}: {e}")
    return audio_clips

clips = load_audio_from_videos("/Users/jeffzhu/Desktop/itau-group4/data/temp_video_extracted/AV1M/extracted/train/lrs3")
print(len(clips), "audio clips loaded")


def prepare_for_vgg16(S_db):
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    S_img = (S_norm * 255).astype(np.uint8)
    S_rgb = np.stack([S_img]*3, axis=-1)
    S_resized = cv2.resize(S_rgb, (224, 224))
    return S_resized
