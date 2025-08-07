import os
import sys
import cv2
import torch
import pickle
import numpy as np
from facenet_pytorch import MTCNN
from moviepy import VideoFileClip
from dropbox.dropbox_utils import download_file

class SenetEmbedder:
    def __init__(self):
        self.model_name = "senet"
        self.mode = "video"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = self.load_model()

        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.MEAN_BGR = np.array([131.0912, 103.8827, 91.4953], dtype=np.float32)

    def get_model_weights(self):
        weight_path = "models/pretrained/senet/senet50_ft_weight.pkl"
        if not os.path.exists(weight_path):
            download_file(weight_path, weight_path)
        
        # Properly load using pickle with latin1 encoding
        with open(weight_path, "rb") as f:
            state = pickle.load(f, encoding="latin1")
        
        # Convert numpy arrays to torch tensors
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                state[k] = torch.from_numpy(v)
        
        return state

    def load_model(self):
        from models.pretrained.senet.senet import senet50
        model = senet50(num_classes=8631, include_top=False).to(self.device)
        model.load_state_dict(self.get_model_weights())
        model.eval()
        return model

    def preprocess_face(self, bgr: np.ndarray) -> torch.Tensor:
        crop = cv2.resize(bgr, (224, 224))
        arr = crop.astype(np.float32) - self.MEAN_BGR
        return torch.from_numpy(arr.transpose(2, 0, 1))

    def embed(self, video_clip: VideoFileClip, pool: bool = True) -> np.ndarray:
        """
        Extract SE-ResNet-50 (VGGFace2-FT) embeddings.
        Returns either mean pooled (2048,) or per-frame (N, 2048) embedding.
        """
        any_face_detected = False
        for frame in video_clip.iter_frames(fps=video_clip.fps, dtype="uint8"):
            face_t = self.mtcnn(frame)
            if face_t is not None:
                any_face_detected = True
                break

        feats = []
        frame_count = 0

        for frame in video_clip.iter_frames(fps=video_clip.fps, dtype="uint8"):
            frame_count += 1
            try:
                if any_face_detected:
                    face_t = self.mtcnn(frame)
                    if face_t is None:
                        continue
                    face_np = (face_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                else:
                    face_np = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                x = self.preprocess_face(face_np).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.model(x)
                    feat = out.view(out.size(0), -1)
                feats.append(feat.squeeze(0).cpu().numpy())

            except Exception as e:
                print(f"⚠️ Frame {frame_count} error: {str(e)}")
                continue

        if not feats:
            print("⚠️ No valid frames processed - returning zero vector")
            return np.zeros(2048) if pool else np.zeros((1, 2048))

        arr = np.stack(feats, axis=0)
        return arr.mean(axis=0) if pool else arr
