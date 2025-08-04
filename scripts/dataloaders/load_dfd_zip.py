import os
import subprocess
from utils.config_loader import load_config
import json
import zipfile

config = load_config()
KG_USER = config["kaggle"]["id"]
KG_KEY = config["kaggle"]["key"]

# Set Kaggle API credentials as environment variables
os.environ["KAGGLE_USERNAME"] = KG_USER
os.environ["KAGGLE_KEY"] = KG_KEY

# Define local storage path
LOCAL_DIR = "./data/temp_video_extracted/DFD"
os.makedirs(LOCAL_DIR, exist_ok=True)

# # Download the dataset using the Kaggle CLI
# subprocess.run([
#     "kaggle", "datasets", "download",
#     "-d", "sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset",
#     "-p", LOCAL_DIR,
#     "--unzip"  # ✅ Unzips automatically
# ])

# print(f"Dataset downloaded and extracted to: {LOCAL_DIR}")

# Step 1: Download full dataset
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", "sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset",
    "-p", LOCAL_DIR
])

# Step 2: Locate ZIP file
zip_files = [fname for fname in os.listdir(LOCAL_DIR) if fname.endswith(".zip")]
assert zip_files, "ZIP not found"
zip_path = os.path.join(LOCAL_DIR, zip_files[0])

EXTRACT_DIR = os.path.join(LOCAL_DIR, "first10")
os.makedirs(EXTRACT_DIR, exist_ok=True)

# Step 3: Extract only first 10 video files
with zipfile.ZipFile(zip_path, "r") as z:
    vids = [f for f in z.namelist() if f.lower().endswith((".mp4", ".avi"))]
    for f in vids[:10]:
        z.extract(f, EXTRACT_DIR)

print(f"Extracted {min(10, len(vids))} videos to: {EXTRACT_DIR}")