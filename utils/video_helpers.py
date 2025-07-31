from moviepy import VideoFileClip
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