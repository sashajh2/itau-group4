import pandas as pd  # type: ignore
from typing import Callable, Any, Optional

class Embedder:
    def __init__(self, model_name: str, checkpoint_path: Optional[str] = None, device: str = 'cpu'):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = self._load_model()

    def _load_model(self) -> Any:
        """
        Placeholder for model loading logic. Should be overridden by subclasses.
        """
        return None

    def embed_dataset(self, df: pd.DataFrame, embedding_fn: Callable, mode: str = 'audio') -> pd.DataFrame:
        """
        Args:
            df (pd.DataFrame): DataFrame with relevant columns for the mode.
            embedding_fn (function): Function that takes in a clip and returns an embedding.
            mode (str): 'audio', 'video', or 'forensic'.
        Returns:
            pd.DataFrame: DataFrame with columns ['embedding', 'label']
        """
        results = []
        for i, row in df.iterrows():
            try:
                if mode == 'video':
                    from moviepy.editor import VideoFileClip  # type: ignore
                    video = VideoFileClip(row['video_path']).subclip(row['segment_start'], row['segment_end'])
                    emb = embedding_fn(video)
                    label = 1 if row.get('video_label', '') == 'fake' else 0
                elif mode == 'audio':
                    from moviepy.editor import VideoFileClip  # type: ignore
                    video = VideoFileClip(row['video_path']).subclip(row['segment_start'], row['segment_end'])
                    audio = video.audio
                    emb = embedding_fn(audio)
                    label = 1 if row.get('audio_label', '') == 'fake' else 0
                elif mode == 'forensic':
                    # Forensic mode: user should define what to pass to embedding_fn
                    emb = embedding_fn(row)
                    label = row.get('label', 0)
                else:
                    raise ValueError("Mode must be 'video', 'audio', or 'forensic'.")
                results.append({'embedding': emb, 'label': label})
            except Exception as e:
                print(f"Skipping row {i} due to error: {e}")
                continue
        return pd.DataFrame(results) 