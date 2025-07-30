import yaml
import os

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        raw = f.read()
    expanded = os.path.expandvars(raw)
    return yaml.safe_load(expanded)

### Example Usage in a script: 
# from utils.config_loader import load_config

# config = load_config()
# dropbox_token = config['dropbox']['access_token']
