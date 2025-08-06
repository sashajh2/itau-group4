from moviepy import VideoFileClip, AudioClip
import numpy as np
import random
from utils import install_ffmpeg
import cv2
import subprocess

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

def extract_three_frames(video_path, start_time, end_time):
    """
    Extracts 3 equally spaced frames from the given time interval of a video.

    Args:
        video_path (str): Path to the video file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.

    Returns:
        List of np.ndarray frames.
    """
    ### TO CHANGE ###
    # Install ffmpeg if necessary
    install_ffmpeg()

    duration = end_time - start_time
    if duration <= 0:
        raise ValueError("End time must be greater than start time.")

    # Generate 3 equally spaced timestamps
    offsets = [0.25, 0.5, 0.75]
    timestamps = [start_time + duration * frac for frac in offsets]
    print(f'Extracting frames at times: {timestamps}')

    # Get video resolution using OpenCV
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frame_size = width * height * 3
    frames = []

    for t in timestamps:
        ts = f"{int(t // 3600):02}:{int((t % 3600) // 60):02}:{t % 60:06.3f}"

        cmd = [
            'ffmpeg', '-ss', ts, '-i', video_path,
            '-frames:v', '1',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-loglevel', 'error',
            '-'
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_frame = process.stdout.read(frame_size)
        process.stdout.close()
        process.wait()

        if len(raw_frame) == frame_size:
            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
            frames.append(frame)
        else:
            print(f"⚠️ Could not extract frame at {ts}")

    return frames