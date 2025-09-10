#Wrapper around Dropbox SDK to abstract upload, download, and access-token logic. Helps avoid copy-pasting Dropbox boilerplate throughout your code.
import dropbox
from utils.config_loader import load_config

_config = load_config()
_dbox_config = _config['dropbox']

APP_KEY = _dbox_config['app_key']
APP_SECRET = _dbox_config['app_secret']
REFRESH_TOKEN = _dbox_config['refresh_token']

_dbx = dropbox.Dropbox(
    app_key=APP_KEY,
    app_secret=APP_SECRET,
    oauth2_refresh_token=REFRESH_TOKEN,
    timeout=300
)

def get_client():
    return _dbx

def upload_file(local_path, dropbox_path, overwrite=True):
    mode = dropbox.files.WriteMode.overwrite if overwrite else dropbox.files.WriteMode.add
    with open(local_path, 'rb') as f:
        data = f.read()
    try:
        get_client().files_upload(data, dropbox_path, mode=mode)
        print(f"File uploaded successfully to {dropbox_path}")
    except dropbox.exceptions.ApiError as e:
        print(f"Error uploading file: {e}")

def download_file(dropbox_path, local_path):
    try:
        metadata = get_client().files_download_to_file(local_path, dropbox_path)
        print(f"File downloaded successfully to {local_path}")
        return metadata
    except dropbox.files.DownloadError as e:
        print(f"Error downloading file: {e}")
        return None

def test_conection():
    try:
        account = get_client().users_get_current_account()
        print(f"Connected to Dropbox as {account.name.display_name}")
        return True
    except dropbox.exceptions.ApiError as e:
        print(f"Error connecting to Dropbox: {e}")
        return False
