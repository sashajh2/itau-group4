import numpy as np
import torch
import torchvision.transforms as transforms
from embedding.utils import extract_three_frames
import dlib
import cv2
import numpy as np

class Args:
    def __init__(self):
        self.n_feats = 64
        self.rgb_range = 255
        self.reduction = 16

class RidnetEmbedder:
    def __init__(self):
        self.model_name = "ridnet"
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

            except Exception as e:
                print(f"Skipping frame due to error: {e}")
                continue

        mean_emb = np.mean(embeddings, axis=0)
        return mean_emb

    def isolate_face(img_array):
        detector = dlib.get_frontal_face_detector()

        # Load image
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Detect faces
        dlib_rects = detector(gray, 1)  # second arg = upsample times

        if not dlib_rects:
            print("No faces detected!")
            return img_array

        # Extract faces and ADD PADDING
        for rect in dlib_rects:
            x1 = rect.left()
            y1 = rect.top()
            x2 = rect.right()
            y2 = rect.bottom()

            # Crop and store face
            face = img_array[y1:y2, x1:x2]

            # Add padding to make it 224x224 if needed
            top, bottom, left, right = 47, 48, 47, 48
            padded = np.pad(face, ((top, bottom), (left, right), (0, 0)),
           mode='constant', constant_values=0)  # -> (224,224,3)

            return padded

    def get_video_face_noise(self, video: str) -> str:
        pass  # Change this to get numpy arrays not strings3

    # def embed(self, video: str, num_components: int = 128) -> np.ndarray:
    #     NOT SURE WE REALLY NEED THIS