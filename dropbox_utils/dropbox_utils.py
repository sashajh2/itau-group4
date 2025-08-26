#Wrapper around Dropbox SDK to abstract upload, download, and access-token logic. Helps avoid copy-pasting Dropbox boilerplate throughout your code.
import dropbox as dropbox_sdk
from utils.config_loader import load_config

config = load_config()

ACCESS_TOKEN = config['dropbox']['access_token']

dbx = dropbox_sdk.Dropbox(ACCESS_TOKEN)

def upload_file(local_path, dropbox_path, overwrite=True):
    mode = dropbox_sdk.files.WriteMode.overwrite if overwrite else dropbox_sdk.files.WriteMode.add
    with open(local_path, 'rb') as f:
        data = f.read()
    try:
        dbx.files_upload(data, dropbox_path, mode=mode)
        print(f"File uploaded successfully to {dropbox_path}")
    except dropbox_sdk.exceptions.ApiError as e:
        print(f"Error uploading file: {e}")

def download_file(dropbox_path, local_path):
    try:
        metadata = dbx.files_download_to_file(local_path, dropbox_path)
        print(f"File downloaded successfully to {local_path}")
        return metadata
    except dropbox_sdk.files.DownloadError as e:
        print(f"Error downloading file: {e}")
        return None

def test_conection():
    try:
        account = dbx.users_get_current_account()
        print(f"Connected to Dropbox as {account.name.display_name}")
        return True
    except dropbox_sdk.exceptions.ApiError as e:
        print(f"Error connecting to Dropbox: {e}")
        return False
