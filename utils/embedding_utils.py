from moviepy import VideoFileClip, AudioClip
import numpy as np
import random

def get_video_duration(video_path):
    try:
        with VideoFileClip(video_path) as video:
            return video.duration  # seconds as float
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        return None


def sample_real_segment(real_duration, segment_length):
    if real_duration <= segment_length:
        return 0.0, min(real_duration, segment_length)
    start = random.uniform(0, real_duration - segment_length)
    return round(start, 2), round(start + segment_length, 2)


def get_audio_array(audio_clip, sr=44100) -> np.ndarray:
    """
    Converts a moviepy AudioClip into a 1D mono NumPy array at the desired sample rate.
    """
    try:
        audio = audio_clip.to_soundarray(fps=sr)
    except Exception as e:
        print(f"⚠️ to_soundarray failed: {e}. Falling back to iter_frames.")
        frames = []
        try:
            for frame in audio_clip.iter_frames(fps=sr, dtype="float32"):
                frames.append(np.reshape(frame, (1, -1)))  # shape: (1, channels)
            if not frames:
                raise ValueError("No frames extracted from fallback.")
            audio = np.vstack(frames)
        except Exception as e:
            raise RuntimeError(f"Fallback audio extraction failed: {e}")

    # Convert stereo to mono if needed
    if audio.ndim == 2 and audio.shape[1] == 2:
        audio = audio.mean(axis=1)

    return audio.astype(np.float32)  # ensure dtype