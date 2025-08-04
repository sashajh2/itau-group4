# import zipfile
# import os
# import gdown

# # Make new directory
# os.makedirs("./data/temp_video_extracted/SynVTA", exist_ok=True)

# url = "https://drive.google.com/drive/folders/1rnlGsMie1Wc3nLXcb5srRAnTbxrv-L0j"
# output = "./data/temp_video_extracted/SynVTA/SynVTA.zip"

# files = gdown.download_folder(url, output=output, quiet=False, use_cookies=False, remaining_ok=True)

# # Output directory
# extract_dir = "./data/temp_video_extracted/SynVTA/extracted"
# os.makedirs(extract_dir, exist_ok=True)

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os

# Authenticate
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Opens browser for auth
drive = GoogleDrive(gauth)

# Target folder ID
folder_id = "1rnlGsMie1Wc3nLXcb5srRAnTbxrv-L0j"

# Query files in folder
file_list = drive.ListFile({
    'q': f"'{folder_id}' in parents and trashed=false"
}).GetList()

print(f"Found {len(file_list)} files")

# Download first 10 video files
video_exts = ('.mp4', '.avi', '.mov', '.mkv') # Extra
os.makedirs("first10_synvta", exist_ok=True)

count = 0
for file in file_list:
    if file['title'].lower().endswith(video_exts):
        print(f"Downloading {file['title']}")
        file.GetContentFile(os.path.join("first10_synvta", file['title']))
        count += 1
        if count == 10:
            break
