import os
import subprocess
from utils.config_loader import load_config
import pandas as pd

# Load Hugging Face token
config = load_config()
HF_TOKEN = config['huggingface']['token']

# Set local output directory and ZIP file to download
LOCAL_DIR = "./data/temp_video_extracted/movieGen"
TABLE_FILENAME = "data/test_with_generations-00000-of-00033.parquet"   # Replace with any file you want
TABLE_FILE = os.path.join(LOCAL_DIR, TABLE_FILENAME)

# Set token as environment variable
os.environ["HF_TOKEN"] = HF_TOKEN

# Login to Hugging Face CLI 
subprocess.run([
    "huggingface-cli", "login", "--token", HF_TOKEN
])

# Download the zip file from the dataset 
subprocess.run([
    "huggingface-cli", "download",
    "meta-ai-for-media-research/movie_gen_video_bench",    # Dataset name (adjust if different)
    TABLE_FILENAME,
    "--repo-type", "dataset",
    "--local-dir", LOCAL_DIR
])

# Load parquet
df = pd.read_parquet(TABLE_FILE)

output_dir = "./data/temp_video_extracted/movieGen"
os.makedirs(output_dir, exist_ok=True)
video_dir= f"{output_dir}/extracted"
os.makedirs(video_dir, exist_ok=True)

file_count = 0
for i, video_bytes in enumerate(df.iloc[:, 1]):
    filename = os.path.join(video_dir, f"video_{i:04d}.mp4")
    with open(filename, "wb") as f:
        f.write(video_bytes)
    print(f"Saved {filename}")
    file_count += 1

print(f"{file_count} Files saved in folder")




