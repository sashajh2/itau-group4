import os
import subprocess
from utils.config_loader import load_config

config = load_config()
HF_TOKEN = config['huggingface']['token']

# Set the local output directory
LOCAL_DIR = "./data/temp_video_extracted/AV1M"
ZIP_FILE = os.path.join(LOCAL_DIR, "train/train.zip.003")

# Set token as environment variable
os.environ["HF_TOKEN"] = HF_TOKEN

# Login to Hugging Face (skip if already logged in)
subprocess.run([
    "huggingface-cli", "login", "--token", HF_TOKEN
])

# Download the .zip.001 file
subprocess.run([
    "huggingface-cli", "download", "ControlNet/AV-Deepfake1M-PlusPlus",
    "train/train.zip.003", "--repo-type", "dataset", "--local-dir", LOCAL_DIR
])

# Extract using 7z (make sure 7z is installed)
subprocess.run(["7z", "x", ZIP_FILE, f"-o{LOCAL_DIR}/extracted"])

