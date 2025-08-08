import numpy as np
import torch
import torchvision.transforms as transforms
from embedding.utils import extract_three_frames

class Args:
    def __init__(self):
        self.n_feats = 64
        self.rgb_range = 255
        self.reduction = 16

class RidnetEmbedder:
    def __init__(self):
        self.model_name = "ridnet_pca_128"
        self.mode = "video noise"
        self.model = self.load_model()  

    def load_model(self):
        from models.pretrained.ridnet.ridnet import RIDNET
        args = Args()
        model = RIDNET(args)
        model.load_state_dict(torch.load("models/pretrained/ridnet/ridnet.pt"))
        model.eval()
        return model
    
    def denoise(self, img):
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
        ])
        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)

        # Clamp and convert to image
        denoised_img = output.squeeze(0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy() # Might need to change this
        return denoised_img

    def get_video_noise(self, video_path: str, start_time: float, end_time: float):
        embeddings = []
        for frame in extract_three_frames(video_path, start_time, end_time):
            try:
                noise = frame - self.denoise(frame)
                emb = noise.flatten()
                embeddings.append(emb)

                # Not sure this is necessary here (outside of colab)
                # # Free memory after each frame
                # del keyframe, noise, emb
                # gc.collect()

            except Exception as e:
                print(f"Skipping frame due to error: {e}")
                continue

        mean_emb = np.mean(embeddings, axis=0)
        return mean_emb

    def get_video_face_noise(self, video: str) -> str:
        pass  # Change this to get numpy arrays not strings

    # def embed(self, video: str, num_components: int = 128) -> np.ndarray:
    #     NOT SURE WE REALLY NEED THIS